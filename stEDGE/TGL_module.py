import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm
from collections import defaultdict
from matplotlib.colors import TwoSlopeNorm
import scanpy as sc
import matplotlib.pyplot as plt
from kneed import KneeLocator

def rank_multiscale_genes_tree(
    adata,
    max_genes=3000,
    lam=0.1,              
    lam_excl=0.1,           
    max_iter=200,
    tol=1e-4,
    step_size=None,
    top_n=50,
    out_dir=None,
    random_state=0,
    verbose=True,
):
    """
    在原 node-split + tree-guided 的基础上，增加轻量 exclusive lasso（兄弟互斥）：
      对每个父节点 v 的兄弟任务组 T(v)，鼓励同一基因不要被多个兄弟 head 同时使用。
    """

    rng = np.random.default_rng(random_state)
    lam_excl =lam_excl*1000
    
    fine_key="fine"
    normal_key="domain"
    coarse_key="domain_coarse"
    domain_map_key="domain_map"
    coarse_map_key="domain_coarse_map"

    excl_fp_iters=2        
    excl_every=2            

    node_weight="sqrt"
    excl_weight="none"

    # ---------- 1) choose expression matrix ----------
    X = adata.X
    var_names = np.array(adata.var_names)

    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

# ---------- 2) subset genes: pick top max_genes by highly_variable_rank ----------
    if (max_genes is not None) and (X.shape[1] > max_genes):
        if "highly_variable_rank" not in adata.var.columns:
            raise KeyError(
                "adata.var 缺少 'highly_variable_rank'。请先运行 HVG 并生成该列。"
            )

        # 对齐到当前 X 的列顺序（var_names 已与 X 对齐）
        v = adata.var.reindex(var_names)

        # rank 越小越好；非HVG通常是 NaN -> 设为 +inf（自然排到最后）
        ranks = v["highly_variable_rank"].astype(float).to_numpy()
        ranks = np.where(np.isfinite(ranks), ranks, np.inf)

        # 取 rank 最小的 max_genes 个（更快：argpartition，再按 rank 排序保证稳定）
        idx0 = np.argpartition(ranks, max_genes - 1)[:max_genes]
        idx0 = idx0[np.argsort(ranks[idx0])]

        X = X[:, idx0]
        var_names = var_names[idx0]

    N, G = X.shape

    # ---------- 3) labels ----------
    if fine_key not in adata.obs:
        raise KeyError(f"adata.obs 缺少 fine_key={fine_key}")
    y_fine = adata.obs[fine_key].astype(str).to_numpy()

    if domain_map_key not in adata.uns or coarse_map_key not in adata.uns:
        raise KeyError(f"adata.uns 需要包含 {domain_map_key} 和 {coarse_map_key}")

    domain_map_raw = adata.uns[domain_map_key]
    coarse_map_raw = adata.uns[coarse_map_key]

    def _s(x): return str(x)
    domain_map = {_s(d): [_s(f) for f in list(fs)] for d, fs in domain_map_raw.items()}
    coarse_map = {_s(c): [_s(d) for d in list(ds)] for c, ds in coarse_map_raw.items()}

    if normal_key in adata.obs:
        y_domain = adata.obs[normal_key].astype(str).to_numpy()
    else:
        fine_to_domain = {}
        for d, fs in domain_map.items():
            for f in fs:
                if f in fine_to_domain and fine_to_domain[f] != d:
                    raise ValueError(f"fine {f} 同时属于 domain {fine_to_domain[f]} 和 {d}")
                fine_to_domain[f] = d
        y_domain = np.array([fine_to_domain.get(f, "NA") for f in y_fine], dtype=object).astype(str)

    if coarse_key in adata.obs:
        y_coarse = adata.obs[coarse_key].astype(str).to_numpy()
    else:
        domain_to_coarse = {}
        for c, ds in coarse_map.items():
            for d in ds:
                if d in domain_to_coarse and domain_to_coarse[d] != c:
                    raise ValueError(f"domain {d} 同时属于 coarse {domain_to_coarse[d]} 和 {c}")
                domain_to_coarse[d] = c
        y_coarse = np.array([domain_to_coarse.get(d, "NA") for d in y_domain], dtype=object).astype(str)

    # ---------- 4) build node split tasks ----------
    def _sort_key(x):
        x = str(x)
        return int(x) if x.isdigit() else x

    coarse_ids = sorted(list(coarse_map.keys()), key=_sort_key)

    node_children = {"root": [f"coarse:{c}" for c in coarse_ids]}
    for c in coarse_ids:
        node_children[f"coarse:{c}"] = [f"domain:{d}" for d in coarse_map.get(c, [])]
    for d in domain_map.keys():
        node_children[f"domain:{d}"] = [f"fine:{f}" for f in domain_map.get(d, [])]

    node_samples = {"root": np.arange(N, dtype=int)}
    for c in coarse_ids:
        node_samples[f"coarse:{c}"] = np.where(y_coarse == c)[0].astype(int)
    for d in domain_map.keys():
        node_samples[f"domain:{d}"] = np.where(y_domain == d)[0].astype(int)

    tasks = []
    node_tasks = defaultdict(list)
    edge_ids = []
    task_meta = []

    def _add_tasks_for_node(node_id, level, child_kind):
        idx = node_samples.get(node_id, None)
        if idx is None or idx.size == 0:
            return

        children = node_children.get(node_id, [])
        if len(children) < 2:
            return

        labels = y_coarse if child_kind == "coarse" else (y_domain if child_kind == "domain" else y_fine)
        child_values = [ch.split(":", 1)[1] for ch in children]

        labels_in_scope = labels[idx]
        present = set(np.unique(labels_in_scope).tolist())
        filtered = [(ch, val) for ch, val in zip(children, child_values) if val in present]
        if len(filtered) < 2:
            return

        for child_id, child_val in filtered:
            y = (labels[idx] == child_val).astype(np.float32)
            pos = int(y.sum())
            neg = int(y.size - pos)
            if pos == 0 or neg == 0:
                continue

            t_idx = len(tasks)
            tasks.append((idx, y))
            node_tasks[node_id].append(t_idx)

            edge_id = f"{node_id}->{child_id}"
            edge_ids.append(edge_id)
            task_meta.append(dict(
                task_idx=t_idx, node_id=node_id, child_id=child_id, edge_id=edge_id,
                level=level, n_samples=int(idx.size), pos=pos, neg=neg
            ))

    _add_tasks_for_node("root", 0, "coarse")
    for c in coarse_ids:
        _add_tasks_for_node(f"coarse:{c}", 1, "domain")
    for d in domain_map.keys():
        _add_tasks_for_node(f"domain:{d}", 2, "fine")

    T = len(tasks)
    if T == 0:
        raise RuntimeError("没有构建出任何有效 split task。")

    task_info = pd.DataFrame(task_meta).sort_values(["level", "node_id", "task_idx"]).reset_index(drop=True)

    # ---------- 5) tree groups (subtree tasks) ----------
    def _weight(sz: int, mode: str):
        if mode == "sqrt":
            return float(np.sqrt(sz))
        return 1.0

    internal_nodes = [nid for nid in node_children.keys()
                      if (nid == "root" or nid.startswith("coarse:") or nid.startswith("domain:"))]

    def _collect_subtree_tasks(node_id, memo):
        if node_id in memo:
            return memo[node_id]
        s = set(node_tasks.get(node_id, []))
        for ch in node_children.get(node_id, []):
            if ch.startswith("coarse:") or ch.startswith("domain:"):
                s |= set(_collect_subtree_tasks(ch, memo))
        out = sorted(s)
        memo[node_id] = out
        return out

    memo = {}
    node_group_tasks = {}
    for nid in internal_nodes:
        idxs = _collect_subtree_tasks(nid, memo)
        if len(idxs) > 0:
            node_group_tasks[nid] = np.array(idxs, dtype=int)

    group_nodes_sorted = sorted(node_group_tasks.keys(), key=lambda k: len(node_group_tasks[k]))

    groups = []
    lambdas = []
    for nid in group_nodes_sorted:
        idx = node_group_tasks[nid]
        groups.append(idx)
        lambdas.append(float(lam * _weight(len(idx), node_weight)))

    # ---------- 5b) sibling groups for exclusive ----------
    sibling_groups = []
    sibling_lams = []
    if lam_excl > 0:
        for nid, tlist in node_tasks.items():
            if len(tlist) >= 2:
                idx = np.array(tlist, dtype=int)
                sibling_groups.append(idx)
                sibling_lams.append(float(lam_excl * _weight(len(idx), excl_weight)))

    # ---------- 6) step size ----------
    if step_size is None:
        if sparse.issparse(X):
            smax = svds(X, k=1, return_singular_vectors=False)[0]
            L = float(smax * smax)
        else:
            v = rng.normal(size=(G,))
            v /= np.linalg.norm(v) + 1e-12
            for _ in range(25):
                v = X.T @ (X @ v)
                v /= np.linalg.norm(v) + 1e-12
            L = float(v @ (X.T @ (X @ v)))
        step_size = 1.0 / (L + 1e-12)

    print("L≈", L, "step_size≈", step_size, "step*lam≈", step_size*lam)

    # ---------- 7) optimize W ----------
    W = np.zeros((T, G), dtype=np.float32)

    history = []

    # ✅ 关键1：tree prox 向量化（完全等价，但避免 per-gene 循环）
    def _tree_prox_inplace(Wmat):
        for idx, lam_node in zip(groups, lambdas):
            thr = step_size * lam_node
            sub = Wmat[idx, :]                            # (m x G)
            norms = np.sqrt((sub * sub).sum(axis=0))      # (G,)
            scale = np.maximum(0.0, 1.0 - thr / (norms + 1e-12)).astype(np.float32)
            Wmat[idx, :] = sub * scale[None, :]
        return Wmat

    # ✅ 关键2：exclusive prox 的“快版”（不排序、全向量化）
    def _exclusive_prox_inplace(Wmat):
        # prox of alpha*(||x||_1)^2  per gene-column, for each sibling group
        for idx, lam_sib in zip(sibling_groups, sibling_lams):
            alpha = step_size * lam_sib
            if alpha <= 0:
                continue
            sub = Wmat[idx, :]                 # (m x G)
            abs_sub = np.abs(sub)
            m = abs_sub.shape[0]

            # init c (assume all active)
            c = (2.0 * alpha * abs_sub.sum(axis=0)) / (1.0 + 2.0 * alpha * m)

            # fixed-point refine (1~2 iters)
            for _ in range(max(0, int(excl_fp_iters))):
                mask = abs_sub > c
                k = mask.sum(axis=0)
                S = (abs_sub * mask).sum(axis=0)
                denom = 1.0 + 2.0 * alpha * k
                c = np.where(k > 0, (2.0 * alpha * S) / (denom + 1e-12), np.inf)

            Wmat[idx, :] = (np.sign(sub) * np.maximum(abs_sub - c, 0.0)).astype(np.float32)
        return Wmat

    def _compute_loss(W_):
        loss = 0.0
        for t, (idx, y) in enumerate(tasks):
            if idx.size == 0:
                continue
            if sparse.issparse(X):
                Xt = X[idx]
                r = (Xt @ W_[t].T) - y
            else:
                Xt = X[idx, :]
                r = (Xt @ W_[t]) - y
            loss += 0.5 * float((r * r).sum())
        return loss

    def _compute_reg_tree(W_):
        reg = 0.0
        for idx, lam_node in zip(groups, lambdas):
            reg += float(lam_node) * float(np.linalg.norm(W_[idx, :], axis=0).sum())
        return reg

    def _compute_reg_excl(W_):
        if lam_excl <= 0 or len(sibling_groups) == 0:
            return 0.0
        reg = 0.0
        for idx, lam_sib in zip(sibling_groups, sibling_lams):
            s = np.abs(W_[idx, :]).sum(axis=0)     # (G,)
            reg += float(lam_sib) * float((s * s).sum())
        return reg

    prev_obj = None
    iters = tqdm(range(1, max_iter + 1), desc="node-split (tree + excl) optimize") if verbose else range(1, max_iter + 1)

    for it in iters:
        grad = np.zeros_like(W, dtype=np.float32)

        for t, (idx, y) in enumerate(tasks):
            if idx.size == 0:
                continue
            if sparse.issparse(X):
                Xt = X[idx]
                r = (Xt @ W[t].T) - y
                g = Xt.T @ r
                grad[t, :] = np.asarray(g).ravel().astype(np.float32)
            else:
                Xt = X[idx, :]
                r = (Xt @ W[t]) - y
                grad[t, :] = (Xt.T @ r).astype(np.float32)

        W -= step_size * grad

        _tree_prox_inplace(W)

        # ✅ 互斥 prox：可选降低频率加速
        if lam_excl > 0 and len(sibling_groups) > 0 and (it % max(1, int(excl_every)) == 0):
            _exclusive_prox_inplace(W)

        # if it % 5 == 0 or it == max_iter:
        #     loss = _compute_loss(W)
        #     reg = _compute_reg_tree(W) + _compute_reg_excl(W)
        #     obj = loss + reg
        #     if prev_obj is not None:
        #         rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
        #         if verbose and hasattr(iters, "set_postfix"):
        #             iters.set_postfix({"obj": f"{obj:.3e}", "rel": f"{rel:.2e}", "T": T})
        #         if rel < tol:
        #             break
        #     prev_obj = obj

        if it % 5 == 0 or it == max_iter:
            loss = _compute_loss(W)
            reg_tree = _compute_reg_tree(W)
            reg_excl = _compute_reg_excl(W)
            obj = loss + reg_tree + reg_excl

            rel = None
            if prev_obj is not None:
                rel = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)

            # ✅ NEW: record
            history.append(dict(
                it=int(it),
                obj=float(obj),
                rel=float(rel) if rel is not None else np.nan,
                loss=float(loss),
                reg_tree=float(reg_tree),
                reg_excl=float(reg_excl),
            ))

            if verbose and hasattr(iters, "set_postfix"):
                if rel is None:
                    iters.set_postfix({"obj": f"{obj:.3e}", "T": T})
                else:
                    iters.set_postfix({"obj": f"{obj:.3e}", "rel": f"{rel:.2e}", "T": T})

            if rel is not None and rel < tol:
                break

            prev_obj = obj


    # ---------- 8) rankings ----------
    global_score = np.linalg.norm(W, axis=0)
    df_global = (
        pd.DataFrame({"gene": var_names, "global_score": global_score})
        .sort_values("global_score", ascending=False)
        .reset_index(drop=True)
    )

    node_rank = {}
    for node_id, tlist in node_tasks.items():
        if len(tlist) == 0:
            continue
        idx = np.array(tlist, dtype=int)
        score = np.linalg.norm(W[idx, :], axis=0)
        node_rank[node_id] = (
            pd.DataFrame({"gene": var_names, "score": score})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
            .head(top_n)
        )

    edge_rank = {}
    edge_rank_by_child = {}
    child_to_parent = {}
    for parent, children in node_children.items():
        for ch in children:
            child_to_parent[ch] = parent

    for t in range(T):
        w = W[t, :].astype(float)                 # ✅ 保留符号
        abs_score = np.abs(w)

        df_t = (
            pd.DataFrame({
                "gene": var_names,
                "weight": w,                      # ✅ 带符号
                "abs_score": abs_score,           # ✅ 用它排序
                "direction": np.where(w >= 0, "pos", "neg")
            })
            .sort_values("abs_score", ascending=False)
            .reset_index(drop=True)
            .head(top_n)
        )

        edge_id = edge_ids[t]
        edge_rank[edge_id] = df_t

        parent, child = edge_id.split("->", 1)
        edge_rank_by_child[child] = df_t

    all_potential_children = list(child_to_parent.keys())

    for child in all_potential_children:
        if child not in edge_rank_by_child:
            # 如果这个 child 没有直接权重，说明它是单分支产生的
            # 向上回溯寻找最近的一个具有有效权重的祖先节点
            curr = child
            while curr in child_to_parent:
                parent = child_to_parent[curr]
                if parent in edge_rank_by_child:
                    # 让该子节点继承父节点的权重
                    edge_rank_by_child[child] = edge_rank_by_child[parent]
                    edge_rank[f"{parent}->{child}"] = edge_rank_by_child[parent]
                    break
                curr = parent


    # ---------- 9) optional save ----------
    gene_tree = {
            "global": df_global.head(top_n),
            "node_rank": node_rank,
            "edge_rank": edge_rank,
            'edge_rank_bc': edge_rank_by_child,
            "W": W,
            "task_info": task_info,
            "groups": groups,
            "lambdas": lambdas,
            "lam_excl": lam_excl,
            "sibling_groups_n": len(sibling_groups),
            "history": pd.DataFrame(history),
        }
    
    if out_dir is not None:
        out_dir = os.path.join(out_dir, "tree_guided_genes_rank")
        os.makedirs(out_dir, exist_ok=True)

        df_global.head(top_n).to_csv(os.path.join(out_dir, "genes_global.csv"), index=False)

        node_dir = os.path.join(out_dir, "node_rank")
        os.makedirs(node_dir, exist_ok=True)
        for nid, df in node_rank.items():
            safe = nid.replace(":", "_")
            df.to_csv(os.path.join(node_dir, f"genes_{safe}.csv"), index=False)

        edge_dir = os.path.join(out_dir, "edge_rank")
        os.makedirs(edge_dir, exist_ok=True)
        for eid, df in edge_rank.items():
            safe = eid.replace(":", "_").replace("->", "__to__").replace("|", "_")
            df.to_csv(os.path.join(edge_dir, f"genes_{safe}.csv"), index=False)

        task_info.to_csv(os.path.join(out_dir, "task_info.csv"), index=False)

        tree_save_path = os.path.join(out_dir, "gene_tree.pkl")
        with open(tree_save_path, 'wb') as f:
            pickle.dump(gene_tree, f)
        print(f"Gene tree dictionary has been saved to: {tree_save_path}")

    return gene_tree



#############################################################################################

import os
import seaborn as sns
import matplotlib.pyplot as plt


def plot_edge_top_genes_gradient(
    gene_tree,
    edge_id,
    top_n=20,
    figure=(8, 6),
    out_dir=None,
    key="edge_rank_bc",
    # 使用红蓝对称色盘 RdBu_r (Red-Blue reversed)
    cmap_name="RdBu_r", 
    dpi=600,
    show=True,
):
    if key not in gene_tree:
        raise KeyError(f"gene_tree 缺少 key='{key}'")

    if edge_id not in gene_tree[key]:
        raise KeyError(f"edge_id='{edge_id}' 不存在")

    df = gene_tree[key][edge_id]
    
    # 1. 依然基于 abs_score 排序并取 Top N
    sub = df.sort_values("abs_score", ascending=False).head(int(top_n)).copy()
    # sub = sub.iloc[::-1] # 反转使最显著的在上方

    # 2. 创建渐变颜色映射
    # TwoSlopeNorm 确保 0 点始终对应色盘的中心颜色（通常是白色/浅色）
    v_min, v_max = sub["weight"].min(), sub["weight"].max()
    # 防止全为正或全为负导致 Norm 报错
    if v_min >= 0: v_min = -0.0001
    if v_max <= 0: v_max = 0.0001
    norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
    cmap = plt.get_cmap(cmap_name)
    color_list = [cmap(norm(x)) for x in sub["weight"]]

    # ---- styling ----
    sns.set_theme(style="white")
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams['axes.unicode_minus'] = False 

    plt.figure(figsize=figure)

    # 绘制 Bar
    ax = sns.barplot(
        x="weight",
        y="gene",
        data=sub,
        palette=color_list, 
        hue="gene",
        legend=False,
        edgecolor="0.3",
        linewidth=0.8,
        alpha=0.9,
    )

    # 核心：添加 0 位垂直参考线
    ax.axvline(0, color='black', linewidth=1, zorder=3)

    sns.despine(left=False, bottom=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(f"Discriminative Gene Features: {edge_id}", fontsize=15, weight="bold", pad=25)
    ax.set_xlabel("Weight", fontsize=12, labelpad=10)
    ax.set_ylabel("")
    

    # 调整 Y 轴基因名样式
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_style("italic")

    # ---- 在绘制 bar 之后添加以下逻辑 ----

    # 1. 计算绝对值最大的分值，用于确定坐标轴范围
    max_abs_score = sub["abs_score"].max()
    
    # 2. 核心：设置对称的 X 轴范围，并留出约 30% 的空白缓冲区给 Label
    # 这样即使 weight 很小，左侧也会有足够的空间显示 "-0.0045"
    buffer_factor = 1
    ax.set_xlim(-max_abs_score * buffer_factor, max_abs_score * buffer_factor)

    # ---- 绘制完 bar 后的动态坐标轴调整 ----

    # 1. 分别获取正负权重的极值
    v_min = sub["weight"].min()
    v_max = sub["weight"].max()

    # 2. 为左右两边分别设置缓冲区 (factor=1.2 表示留出 20% 的空间放标注)
    # 对于负数区间：
    if v_min < 0:
        x_left = v_min * 1.3  # 负数乘以 1.3 会变得更小，即向左延伸更多空间
    else:
        x_left = -0.0001       # 如果全是正数，给左边留一点点缝隙
        
    # 对于正数区间：
    if v_max > 0:
        x_right = v_max * 1.3 # 正数向右延伸空间
    else:
        x_right = 0.001       # 如果全是负数，给右边留一点点缝隙

    # 3. 设置非对称 X 轴
    ax.set_xlim(x_left, x_right)

    # 4. 强制显示 0 轴线，防止它因为非对称而显得突兀
    ax.axvline(0, color='black', linewidth=1.2, zorder=3)

    # 5. 标注逻辑保持不变，因为它已经能自适应 ha='left' 或 'right'
    for p in ax.patches:
        width = p.get_width()
        ha = 'left' if width > 0 else 'right'
        offset = 6 if width > 0 else -6
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2),
            ha=ha, va="center",
            xytext=(offset, 0),
            textcoords="offset points",
            fontsize=9, fontweight="semibold"
        )
        
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe_edge = edge_id.replace(":", "_").replace("->", "__to__")
        plt.savefig(os.path.join(out_dir, f"gradient_bar_{safe_edge}.png"), dpi=dpi, bbox_inches="tight")

    if show: plt.show()
    else: plt.close()



def calculate_edge_scores(
    adata, 
    gene_tree, 
    edge_id, 
    res_key="edge_rank_bc",
    Knee_S=1,
    min_genes=10,
    max_genes=100,
    return_gene_list=False
):
    """
    计算给定 edge_id 的加权表达评分。
    逻辑：动态 N 选择 -> Sum(Pos) - Sum(Neg)
    
    返回:
        np.ndarray: 每个 spot 的评分数组 (shape: n_obs,)
        (可选) dict: 包含选中的正向和负向基因列表
    """
    
    # 1. 获取并排序权重数据
    if edge_id not in gene_tree[res_key]:
        raise KeyError(f"Edge {edge_id} not found in {res_key}")
        
    df_edge = gene_tree[res_key][edge_id].copy()
    df_edge = df_edge.sort_values("abs_score", ascending=False).reset_index(drop=True)
    
    # 2. 动态寻找肘点确定基因数量
    scores = df_edge["abs_score"].values
    x = np.arange(len(scores))
    kn = KneeLocator(x, scores, curve='convex', direction='decreasing',interp_method='interp1d', S=Knee_S)
    knee_idx = kn.knee
    if knee_idx is None or knee_idx == 0:
        kn_poly = KneeLocator(x, scores, curve='convex', direction='decreasing',interp_method='polynomial', S=Knee_S)
        knee_idx = kn_poly.knee
    if knee_idx is None or knee_idx < 2:
            knee_idx = min_genes

    dynamic_n = int(np.clip(knee_idx, min_genes, max_genes))
    
    # 3. 提取 top 基因
    top_genes_df = df_edge.head(dynamic_n)
    pos_genes = [g for g in top_genes_df[top_genes_df['direction'].str.contains('pos')]['gene'] if g in adata.var_names]
    neg_genes = [g for g in top_genes_df[top_genes_df['direction'].str.contains('neg')]['gene'] if g in adata.var_names]
    
    # 4. 矩阵累加计算 (Sum(Pos) - Sum(Neg))
    final_scores = np.zeros(adata.n_obs)

    if pos_genes:
        X_pos = adata[:, pos_genes].X
        if hasattr(X_pos, "toarray"): X_pos = X_pos.toarray()
        final_scores += X_pos.sum(axis=1).flatten()

    if neg_genes:
        X_neg = adata[:, neg_genes].X
        if hasattr(X_neg, "toarray"): X_neg = X_neg.toarray()
        final_scores -= X_neg.sum(axis=1).flatten()
        
    if return_gene_list:
        return final_scores, {"pos": pos_genes, "neg": neg_genes, "n": dynamic_n}
    
    return final_scores

import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

def plot_gene_weights_spatial(
    adata, 
    gene_tree, 
    edge_id, 
    res_key="edge_rank_bc",
    out_dir=None,
    spot_size=100,
    Knee_S=1,
    show_knee_plot=True,
    show=True,
    cmap='RdYlBu_r',
    **kwargs
):
    """
    通过调用 calculate_edge_scores 获取数值，并进行空间可视化。
    """
    
    # 1. 调用计算函数获取 final_scores 和基因信息
    # 我们需要 return_gene_list=True 来获取 dynamic_n 用于画肘点图
    final_scores, info = calculate_edge_scores(
        adata=adata,
        gene_tree=gene_tree,
        edge_id=edge_id,
        res_key=res_key,
        Knee_S=Knee_S,
        return_gene_list=True
    )
    
    # 2. 准备安全的文件名和数据列名
    safe_edge = edge_id.replace(':', '_')
    score_name = f"{safe_edge}"
    adata.obs[score_name] = final_scores

    # 3. 诊断图绘制 (Knee Plot)
    # 注意：calculate_edge_scores 内部已经完成了 KneeLocator 的逻辑
    # 我们直接利用它返回的 dynamic_n 和原始数据绘图
    if show_knee_plot or out_dir:
        df_edge = gene_tree[res_key][edge_id].sort_values("abs_score", ascending=False)
        scores_val = df_edge["abs_score"].values
        
        plt.figure(figsize=(5, 4))
        plt.plot(range(len(scores_val)), scores_val, 'b-', label='Contribution Score')
        plt.axvline(info['n'], color='r', linestyle='--', label=f"Knee Point (n={info['n']})")
        plt.title(f"Dynamic N Selection: {edge_id}")
        plt.xlabel("Gene Rank")
        plt.ylabel("Abs Score")
        plt.legend()
        plt.grid(alpha=0.3)
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            knee_path = os.path.join(out_dir, f"knee_plot_{safe_edge}.png")
            plt.savefig(knee_path, dpi=300, bbox_inches='tight')
            
        if show_knee_plot:
            plt.show()
        else:
            plt.close()

    # 4. 空间绘图
    # 使用 Scanpy 绘制
    sc.pl.spatial(
        adata, 
        color=score_name, 
        spot_size=spot_size,
        cmap=cmap, 
        show=False,
        **kwargs
    )

    # 5. 保存空间图
    if out_dir:
        outpath = os.path.join(out_dir, f"spatial_{safe_edge}.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)

    # 6. 显示或释放内存
    if show: 
        plt.show()
    else: 
        plt.close()
        
    return 


# def plot_gene_weights_spatial(
#     adata, 
#     gene_tree, 
#     edge_id, 
#     res_key="edge_rank_bc",
#     out_dir=None,
#     spot_size=100,
#     Knee_S=1,
#     show_knee_plot=True,
#     show=True,
#     cmap='RdYlBu_r',
#     **kwargs
# ):
    
#     min_genes=0   
#     max_genes=100  

#     # 1. 获取并排序基因权重数据
#     df_edge = gene_tree[res_key][edge_id].copy()
#     df_edge = df_edge.sort_values("abs_score", ascending=False).reset_index(drop=True)
    
#     # 2. 寻找肘点 (Knee Detection)
#     scores = df_edge["abs_score"].values
#     x = np.arange(len(scores))
#     kn = KneeLocator(x, scores, curve='convex', direction='decreasing', S=Knee_S)
    
#     knee_idx = kn.knee if kn.knee is not None else min_genes
#     dynamic_n = int(np.clip(knee_idx, min_genes, max_genes))
    
#     # 文件名安全处理 (将 : 替换为 _)
#     safe_edge = edge_id.replace(':', '_')

#     # 3. 诊断图绘制与保存
#     if show_knee_plot or out_dir:
#         plt.figure(figsize=(5, 4))
#         plt.plot(x, scores, 'b-', label='Contribution Score')
#         plt.axvline(dynamic_n, color='r', linestyle='--', label=f'Knee Point (n={dynamic_n})')
#         plt.title(f"Dynamic N Selection: {edge_id}")
#         plt.xlabel("Gene Rank")
#         plt.ylabel("Abs Score")
#         plt.legend()
#         plt.grid(alpha=0.3)
        
#         if out_dir:
#             os.makedirs(out_dir, exist_ok=True)
#             knee_path = os.path.join(out_dir, f"knee_plot_{safe_edge}.png")
#             plt.savefig(knee_path, dpi=300, bbox_inches='tight')
#             # print(f"Knee plot saved to: {knee_path}")
            
#         if show_knee_plot:
#             plt.show()
#         else:
#             plt.close()

#     # 4. 根据动态 N 提取基因
#     top_genes_df = df_edge.head(dynamic_n)
#     pos_genes = [g for g in top_genes_df[top_genes_df['direction'].str.contains('pos')]['gene'] if g in adata.var_names]
#     neg_genes = [g for g in top_genes_df[top_genes_df['direction'].str.contains('neg')]['gene'] if g in adata.var_names]
    
#     # print(f">>> Edge {edge_id}: Selected {dynamic_n} genes (Pos:{len(pos_genes)}, Neg:{len(neg_genes)})")

#     # 5. 计算评分 (Sum(Pos) - Sum(Neg))
#     final_scores = np.zeros(adata.n_obs)

#     if pos_genes:
#         X_pos = adata[:, pos_genes].X
#         if hasattr(X_pos, "toarray"): X_pos = X_pos.toarray()
#         final_scores += X_pos.sum(axis=1).flatten() # 确保是一维

#     if neg_genes:
#         X_neg = adata[:, neg_genes].X
#         if hasattr(X_neg, "toarray"): X_neg = X_neg.toarray()
#         final_scores -= X_neg.sum(axis=1).flatten()
    
#     # 6. 存入 adata 并绘图
#     score_name = f"{safe_edge}"
#     adata.obs[score_name] = final_scores

#     # 7. 空间绘图与保存
    
#     # 处理保存路径
    
#     sc.pl.spatial(
#         adata, 
#         color=score_name, 
#         spot_size=spot_size,
#         cmap=cmap, 
#         show=False,
#         **kwargs
#     )

#     # 如果 Scanpy 的 save 逻辑不符合你的 out_dir 路径要求，我们可以手动保存：
#     if out_dir:
#         outpath = os.path.join(out_dir, f"spatial_{safe_edge}.png")
#         plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)
#         # plt.close()
#         # print(f"Spatial plot saved to: {outpath}")

#     if show: plt.show()
#     else: plt.close()
#     return 

##############################################################################################

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

def _get_X_and_varnames(adata, use_raw=True, layer=None):
    if use_raw and (adata.raw is not None):
        X = adata.raw.X
        var_names = np.array(adata.raw.var_names)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = np.array(adata.var_names)
    else:
        X = adata.X
        var_names = np.array(adata.var_names)

    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)  
    return X, var_names


def _infer_domain_coarse_if_missing(adata, fine_key, domain_key, coarse_key, domain_map_key, coarse_map_key):
    """如果 adata.obs 缺 domain/coarse，用 uns 的 map 反推。返回 domain_series, coarse_series (str)."""
    fine_series = adata.obs[fine_key].astype(str)

    # domain
    if domain_key in adata.obs:
        domain_series = adata.obs[domain_key].astype(str)
    else:
        domain_map_raw = adata.uns[domain_map_key]  # domain -> [fine]
        domain_map = {str(d): [str(f) for f in list(fs)] for d, fs in domain_map_raw.items()}
        fine_to_domain = {}
        for d, fs in domain_map.items():
            for f in fs:
                if f in fine_to_domain and fine_to_domain[f] != d:
                    raise ValueError(f"fine {f} 同时属于 domain {fine_to_domain[f]} 和 {d}")
                fine_to_domain[f] = d
        domain_series = fine_series.map(fine_to_domain).astype(str)

    # coarse
    if coarse_key in adata.obs:
        coarse_series = adata.obs[coarse_key].astype(str)
    else:
        coarse_map_raw = adata.uns[coarse_map_key]  # coarse -> [domain]
        coarse_map = {str(c): [str(d) for d in list(ds)] for c, ds in coarse_map_raw.items()} 
        domain_to_coarse = {}
        for c, ds in coarse_map.items():
            for d in ds:
                if d in domain_to_coarse and domain_to_coarse[d] != c:
                    raise ValueError(f"domain {d} 同时属于 coarse {domain_to_coarse[d]} 和 {c}")
                domain_to_coarse[d] = c
        coarse_series = domain_series.map(domain_to_coarse).astype(str)

    return domain_series, coarse_series


def _get_scope_idx_for_edge(
    adata, node_id,
    domain_series, coarse_series,
    coarse_map,  # str -> list[str]  (coarse -> domains)
    scope_mode="parent",
):
    N = adata.n_obs
    if scope_mode == "global":
        return np.arange(N, dtype=int)

    if node_id == "root":
        return np.arange(N, dtype=int)

    if node_id.startswith("coarse:"):
        c = node_id.split(":", 1)[1]
        return np.where(coarse_series.values == str(c))[0].astype(int)

    if node_id.startswith("domain:"):
        d = node_id.split(":", 1)[1]
        if scope_mode == "parent":
            return np.where(domain_series.values == str(d))[0].astype(int)

        if scope_mode == "coarse":
            # 找到该 domain 属于哪个 coarse（从 coarse_map 反查）
            d = str(d)
            c_of_d = None
            for c, ds in coarse_map.items():
                if d in [str(x) for x in ds]:
                    c_of_d = str(c)
                    break
            if c_of_d is None:
                raise ValueError(f"domain {d} 在 coarse_map 里找不到归属 coarse")
            return np.where(coarse_series.values == c_of_d)[0].astype(int)

    raise ValueError(f"unsupported node_id={node_id} / scope_mode={scope_mode}")


def _safe_id(s: str) -> str:
    return str(s).replace(":", "_").replace("->", "__to__").replace("/", "_").replace("|", "_")

def compute_edge_discrimination(
    adata,
    out,
    edge_id,
    k=30,
    cv=5,
    random_state=0,
    use_raw=True,
    layer=None,
    standardize=True,
    fine_key="fine",
    domain_key="domain",
    coarse_key="domain_coarse",
    domain_map_key="domain_map",
    coarse_map_key="domain_coarse_map",
    scope_mode="parent",
    rank_key="edge_rank_bc",
    out_dir=None,         # ✅ 新增：保存目录
    prefix=None,          # ✅ 新增：文件名前缀（不填默认用 edge_full_id）
):
    """
    只计算：CV logistic。可选保存 metrics/genes/proba/y。
    """
    # ---------- 0) 取 edge 的 top-k genes ----------
    df_edge = None
    edge_full_id = None

    if rank_key in out and edge_id in out[rank_key]:
        df_edge = out[rank_key][edge_id]
        if isinstance(df_edge, list):
            if len(df_edge) == 0:
                raise KeyError(f"out['{rank_key}'][{edge_id}] 是空 list")
            df_edge = df_edge[0]
        child_id_input = edge_id

    elif "edge_rank" in out and edge_id in out["edge_rank"]:
        df_edge = out["edge_rank"][edge_id]
        edge_full_id = edge_id
        child_id_input = edge_id.split("->", 1)[1]

    else:
        raise KeyError(
            f"找不到 edge_id={edge_id}：既不在 out['{rank_key}']，也不在 out['edge_rank']。"
        )

    genes = df_edge["gene"].head(k).astype(str).tolist()

    # ---------- 1) 解析 node_id / child_id（依赖 task_info） ----------
    if edge_full_id is not None:
        node_id, child_id = edge_full_id.split("->", 1)
    else:
        child_id = str(child_id_input)
        if "task_info" not in out or out["task_info"] is None:
            raise KeyError("out 缺少 task_info，无法从 child_id 反推出 node_id（请保留 task_info 或改用 edge_rank 的 parent->child id）。")
        ti = out["task_info"]
        hit = ti[ti["child_id"].astype(str) == child_id]
        if hit.shape[0] == 0:
            raise KeyError(f"task_info 里找不到 child_id={child_id}（edge_id_input={edge_id}）")
        row = hit.iloc[0]
        node_id = str(row["node_id"])
        edge_full_id = str(row["edge_id"]) if "edge_id" in row.index else f"{node_id}->{child_id}"

    # ---------- 2) labels ----------
    domain_series, coarse_series = _infer_domain_coarse_if_missing(
        adata, fine_key, domain_key, coarse_key, domain_map_key, coarse_map_key
    )
    fine_series = adata.obs[fine_key].astype(str)

    # ---------- 3) coarse_map + scope idx ----------
    coarse_map_raw = adata.uns[coarse_map_key]
    coarse_map = {str(c): [str(d) for d in list(ds)] for c, ds in coarse_map_raw.items()}

    idx = _get_scope_idx_for_edge(
        adata, node_id,
        domain_series, coarse_series,
        coarse_map,
        scope_mode=scope_mode,
    )
    if idx.size < 10:
        raise RuntimeError(f"scope 样本太少：{idx.size}")

    # ---------- 4) y ----------
    if child_id.startswith("coarse:"):
        pos_mask = (coarse_series.values == child_id.split(":", 1)[1])
    elif child_id.startswith("domain:"):
        pos_mask = (domain_series.values == child_id.split(":", 1)[1])
    elif child_id.startswith("fine:"):
        pos_mask = (fine_series.values == child_id.split(":", 1)[1])
    else:
        raise ValueError(f"不支持的 child_id={child_id}")

    y = pos_mask[idx].astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        raise RuntimeError(f"该 edge 在 scope 内退化：pos={y.sum()}, neg={y.size - y.sum()}")

    # ---------- 5) X (top-k genes) ----------
    Xall, var_names = _get_X_and_varnames(adata, use_raw=use_raw, layer=layer)
    g2i = {g: i for i, g in enumerate(var_names)}
    keep_genes = [g for g in genes if g in g2i]
    if len(keep_genes) == 0:
        raise RuntimeError("top-k 基因在当前 var_names 里一个都找不到（检查 use_raw/layer 或基因名）")

    cols = np.array([g2i[g] for g in keep_genes], dtype=int)
    X = Xall[idx][:, cols]
    is_sparse = sparse.issparse(X)

    # ---------- 6) CV proba ----------
    scaler = StandardScaler(with_mean=(standardize and (not is_sparse)))
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = make_pipeline(scaler, clf)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]

    # ---------- 7) metrics ----------
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    thr = 0.5
    yhat = (proba >= thr).astype(int)
    cm = confusion_matrix(y, yhat)

    metrics = {
        "edge_id_input": edge_id,
        "edge_id_full": edge_full_id,
        "node_id": node_id,
        "child_id": child_id,
        "k": len(keep_genes),
        "n_scope": int(idx.size),
        "pos": int(y.sum()),
        "neg": int((y == 0).sum()),
        "roc_auc": float(roc_auc),
        "avg_precision": float(ap),
        "threshold": float(thr),
        "confusion_matrix": cm,  # numpy
        "genes_used": keep_genes,
        "rank_key_used": rank_key if (rank_key in out and edge_id in out.get(rank_key, {})) else "edge_rank",
    }

    result = {
        "metrics": metrics,
        "proba": proba,
        "y": y,
        "keep_genes": keep_genes,
        "idx": idx,
        "edge_full_id": edge_full_id,
    }

    # ---------- 8) optional save (make subfolder; save all as CSV) ----------
    saved = {}
    if out_dir is not None:
        base = prefix if prefix is not None else _safe_id(edge_full_id)
        save_dir = os.path.join(out_dir, base)
        os.makedirs(save_dir, exist_ok=True)

        # 1) metrics.csv
        m = dict(metrics)
        cm = m.pop("confusion_matrix")  # numpy array
        genes_used = m.pop("genes_used")  # list[str]

        # 展平成一行 metrics
        df_metrics = pd.DataFrame([m])
        p_metrics = os.path.join(save_dir, "metrics.csv")
        df_metrics.to_csv(p_metrics, index=False)
        saved["metrics_csv"] = p_metrics

        # 2) confusion_matrix.csv（单独存，避免 metrics 一行里塞复杂结构）
        p_cm = os.path.join(save_dir, "confusion_matrix.csv")
        pd.DataFrame(cm, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"]).to_csv(p_cm)
        saved["confusion_matrix_csv"] = p_cm

        # 3) genes_used.csv
        p_genes = os.path.join(save_dir, "genes_used.csv")
        pd.DataFrame({"gene": genes_used}).to_csv(p_genes, index=False)
        saved["genes_used_csv"] = p_genes

        # 4) proba_y.csv（把 y 和 proba 放一起）
        p_py = os.path.join(save_dir, "proba_y.csv")
        pd.DataFrame({"y": y.astype(int), "proba": proba.astype(float)}).to_csv(p_py, index=False)
        saved["proba_y_csv"] = p_py

        # 5) scope_idx.csv（可选但很有用：记录这次评估用了哪些样本）
        p_idx = os.path.join(save_dir, "scope_idx.csv")
        pd.DataFrame({"idx": idx.astype(int)}).to_csv(p_idx, index=False)
        saved["scope_idx_csv"] = p_idx

        result["saved_compute"] = saved
        return result



def plot_edge_discrimination(
    adata,
    result,
    plot=True,
    spatial_plot=True,
    spot_size=1.3,
    color_map="viridis",
    out_dir=None,        # ✅ 新增：保存目录
    prefix=None,         # ✅ 新增：文件名前缀
    dpi=600,
    show=True,
):
    """
    只作图：ROC/PR/CM/分布 + spatial（临时写入 obs 再删除）。
    可选保存 png。
    """
    metrics = result["metrics"]
    proba = result["proba"]
    y = result["y"]
    idx = result["idx"]
    edge_full_id = result["edge_full_id"]

    roc_auc = metrics["roc_auc"]
    ap = metrics["avg_precision"]
    thr = metrics["threshold"]
    cm = metrics["confusion_matrix"]

    saved = {}

    base = prefix if prefix is not None else _safe_id(edge_full_id)

    if out_dir is not None:
        base = prefix if prefix is not None else _safe_id(edge_full_id)
        save_dir = os.path.join(out_dir, base)
        os.makedirs(save_dir, exist_ok=True)
    

    # --------- 1) 2x2 图 ---------
    if plot:
        fpr, tpr, _ = roc_curve(y, proba)
        prec, rec, _ = precision_recall_curve(y, proba)

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        ax1, ax2, ax3, ax4 = axes.ravel()

        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], linestyle="--")
        ax1.set_title(f"ROC  AUC={roc_auc:.3f}\n{edge_full_id}")
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")

        ax2.plot(rec, prec)
        ax2.set_title(f"PR  AP={ap:.3f}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")

        im = ax3.imshow(cm, cmap="Blues", interpolation="nearest")
        ax3.set_title(f"Confusion Matrix (thr={thr})")
        ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
        ax3.set_xticklabels(["neg", "pos"])
        ax3.set_yticklabels(["neg", "pos"])
        for (i, j), v in np.ndenumerate(cm):
            ax3.text(j, i, str(v), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        ax4.hist(proba[y == 0], bins=25, alpha=0.7, label="neg (siblings)")
        ax4.hist(proba[y == 1], bins=25, alpha=0.7, label="pos (child)")
        ax4.set_title("Predicted probability distribution")
        ax4.set_xlabel("P(pos)")
        ax4.set_ylabel("Count")
        ax4.legend()

        fig.tight_layout()

        if out_dir is not None:
            p_fig = os.path.join(save_dir, f"{base}_roc_pr_cm_hist.png")
            fig.savefig(p_fig, dpi=dpi, bbox_inches="tight")
            saved["eval_png"] = p_fig

        if show:
            plt.show()
        else:
            plt.close(fig)

    # --------- 2) spatial（临时列） ---------
    if spatial_plot:
        tmp = np.full(adata.n_obs, np.nan, dtype=float)
        tmp[idx] = proba

        tmp_col = f"__tmp_edge_prob__{edge_full_id}".replace(":", "_").replace("->", "__to__")
        adata.obs[tmp_col] = tmp

        sc.pl.spatial(
            adata,
            color=tmp_col,
            color_map=color_map,
            spot_size=spot_size,
            frameon=False,
            title=f"{edge_full_id}  (AUC={roc_auc:.3f}, AP={ap:.3f})",
            show=False if (out_dir is not None) or (not show) else True,
        )

        fig_sp = plt.gcf()

        if out_dir is not None:
            p_sp = os.path.join(save_dir, f"{base}_spatial.png")
            fig_sp.savefig(p_sp, dpi=dpi, bbox_inches="tight")
            saved["spatial_png"] = p_sp

        if show:
            plt.show()
        else:
            plt.close(fig_sp)

        del adata.obs[tmp_col]

    result["saved_plot"] = saved
    return saved


def run_edge_discrimination_eval(
    adata,
    out,
    edge_id,
    out_dir=None,
    prefix=None,
    show=True,
    **kwargs,
):
    """
    主函数：先 compute 再 plot。把 out_dir/prefix/save/show 统一管理。
    其余参数透传给 compute / plot（如 k, cv, scope_mode, spot_size...）。
    """
    res = compute_edge_discrimination(
        adata=adata,
        out=out,
        edge_id=edge_id,
        out_dir=out_dir,
        prefix=prefix,
        **{k: v for k, v in kwargs.items()
           if k in [
               "k","cv","random_state","use_raw","layer","standardize",
               "fine_key","domain_key","coarse_key","domain_map_key","coarse_map_key",
               "scope_mode","rank_key"
           ]}
    )

    plot_edge_discrimination(
        adata=adata,
        result=res,
        out_dir=out_dir,
        prefix=prefix,
        show=show,
        **{k: v for k, v in kwargs.items()
           if k in ["plot","spatial_plot","spot_size","color_map","dpi"]}
    )

    return res


import os
import scanpy as sc
from tqdm import tqdm

def batch_plot_gene_tree_results(adata, 
                                 gene_tree, 
                                 base_out_dir,
                                 spot_size=140,
                                 top_n=20,
                                 res_key="edge_rank_bc"):
    """
    自动遍历 gene_tree 中的所有节点，并执行梯度条形图与空间分布图的绘制。
    """
    _adata=adata.copy()
    # 1. 创建基础的 gene_tree 总目录
    main_dir = os.path.join(base_out_dir, "gene_tree")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Created main directory: {main_dir}")

    # 2. 获取所有的 edge_id (即你的 coarse:1, domain:1, fine:1 等)
    all_keys = list(gene_tree[res_key].keys())
    print(f"Total keys found: {len(all_keys)}")

    valid_tasks = set()
    if "task_info" in gene_tree:
        valid_tasks = set(gene_tree["task_info"]["child_id"].unique())

    for edge_id in tqdm(all_keys):
        # 安全处理文件夹名称 (替换冒号)
        safe_key = edge_id.replace(":", "_")
        key_dir = os.path.join(main_dir, safe_key)
        
        if not os.path.exists(key_dir):
            os.makedirs(key_dir)
        
        # print(f"--- Processing: {edge_id} ---")

        # 3. 执行梯度条形图绘制
        # 注意：这里将 out_dir 设为刚刚创建的 key_dir
        try:
            plot_edge_top_genes_gradient(
                gene_tree,
                edge_id=edge_id,
                top_n=top_n,
                figure=(8, 6),
                out_dir=key_dir,
                key=res_key,
                cmap_name="RdBu_r", 
                dpi=600, 
                show=False 
            )
        except Exception as e:
            print(f"Error in gradient plot for {edge_id}: {e}")

        # 4. 执行空间权重分布图绘制
        try:
            plot_gene_weights_spatial(
                _adata, 
                gene_tree, 
                edge_id=edge_id,
                spot_size=spot_size,
                cmap='RdYlBu_r', # 空间图推荐用 RdYlBu_r，中间黄色更易在白色背景识别
                Knee_S=3,
                show_knee_plot=False,
                show=False,
                out_dir=key_dir
            )
        except Exception as e:
            print(f"Error in spatial plot for {edge_id}: {e}")
        
        is_single_child = edge_id not in valid_tasks
        if is_single_child:
            continue
        else:
            run_edge_discrimination_eval(
                _adata,
                out=gene_tree,
                edge_id=edge_id,
                k=100,
                cv=5,
                scope_mode="parent",
                plot=True,
                spatial_plot=True,
                spot_size=spot_size,
                out_dir=key_dir,
                show=False,
            )


    print("\nBatch Processing Completed!")