import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from joblib import Parallel, delayed
from kneed import KneeLocator
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
    use_tree=True,
):
    """
    Tree-guided multiscale discriminative gene discovery with exclusive lasso.

    Builds binary classification tasks for each tree node split (coarse →
    domain → fine), then optimises a coefficient matrix W via proximal gradient
    descent with two structured penalties:

    - **Tree group lasso** (``lam``): subtree tasks share sparsity, promoting
      genes relevant to entire branches.
    - **Exclusive lasso** (``lam_excl``): sibling tasks compete for genes,
      reducing overlap between same-parent edges.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["fine"]``, ``uns["domain_map"]``,
        ``uns["domain_coarse_map"]``, and a log-normalised expression matrix.
    max_genes : int or None
        Number of highly variable genes to retain. If None, all genes are used.
    lam : float
        Tree group-lasso regularisation strength.
    lam_excl : float
        Exclusive-lasso regularisation strength (scaled internally by 1000).
    max_iter : int
        Maximum proximal gradient iterations.
    tol : float
        Convergence tolerance on relative objective change.
    step_size : float or None
        Gradient step size. Auto-estimated from the spectral norm if None.
    top_n : int
        Number of top genes to report in rankings.
    out_dir : str or None
        If set, save gene rankings and the full gene_tree dict to this directory.
    random_state : int
        Random seed for numpy RNG.
    verbose : bool
        Whether to print progress with tqdm.
    use_tree : bool
        If False, all tasks are placed in a single flat group (used for
        ablation ABL7).

    Returns
    -------
    gene_tree : dict
        Contains ``global``, ``node_rank``, ``edge_rank``, ``edge_rank_bc``,
        ``W``, ``task_info``, ``groups``, ``lambdas``, ``history``.
    """

    rng = np.random.default_rng(random_state)
    lam_excl = lam_excl * 1000

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
                "adata.var missing 'highly_variable_rank'. Run HVG first."
            )

        # align to current X column order
        v = adata.var.reindex(var_names)

        # lower rank = better; non-HVG NaN -> +inf
        ranks = v["highly_variable_rank"].astype(float).to_numpy()
        ranks = np.where(np.isfinite(ranks), ranks, np.inf)

        # select top max_genes by rank (argpartition + stable sort)
        idx0 = np.argpartition(ranks, max_genes - 1)[:max_genes]
        idx0 = idx0[np.argsort(ranks[idx0])]

        X = X[:, idx0]
        var_names = var_names[idx0]

    N, G = X.shape

    # ---------- 3) labels ----------
    if fine_key not in adata.obs:
        raise KeyError(f"adata.obs missing {fine_key}")
    y_fine = adata.obs[fine_key].astype(str).to_numpy()

    if domain_map_key not in adata.uns or coarse_map_key not in adata.uns:
        raise KeyError(f"adata.uns must contain {domain_map_key} and {coarse_map_key}")

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
                    raise ValueError(f"fine {f} belongs to both domain {fine_to_domain[f]} and {d}")
                fine_to_domain[f] = d
        y_domain = np.array([fine_to_domain.get(f, "NA") for f in y_fine], dtype=object).astype(str)

    if coarse_key in adata.obs:
        y_coarse = adata.obs[coarse_key].astype(str).to_numpy()
    else:
        domain_to_coarse = {}
        for c, ds in coarse_map.items():
            for d in ds:
                if d in domain_to_coarse and domain_to_coarse[d] != c:
                    raise ValueError(f"domain {d} belongs to both coarse {domain_to_coarse[d]} and {c}")
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
        raise RuntimeError("No valid split tasks were constructed.")

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

    # ---------- 5a) tree groups (or flat single group) ----------
    if use_tree:
        group_nodes_sorted = sorted(node_group_tasks.keys(), key=lambda k: len(node_group_tasks[k]))
        groups = []
        lambdas = []
        for nid in group_nodes_sorted:
            idx = node_group_tasks[nid]
            groups.append(idx)
            lambdas.append(float(lam * _weight(len(idx), node_weight)))
    else:
        # Flat mode: all tasks in one group — no tree hierarchy
        groups = [np.arange(T, dtype=int)]
        lambdas = [float(lam * _weight(T, node_weight))]

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

    # tree prox — vectorized (avoids per-gene loop)
    def _tree_prox_inplace(Wmat):
        for idx, lam_node in zip(groups, lambdas):
            thr = step_size * lam_node
            sub = Wmat[idx, :]                            # (m x G)
            norms = np.sqrt((sub * sub).sum(axis=0))      # (G,)
            scale = np.maximum(0.0, 1.0 - thr / (norms + 1e-12)).astype(np.float32)
            Wmat[idx, :] = sub * scale[None, :]
        return Wmat

    # exclusive prox — fast version (no sorting, fully vectorized)
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

        # exclusive prox: optional decimation for speed
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

            # record history
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
        w = W[t, :].astype(float)  # preserve sign
        abs_score = np.abs(w)

        df_t = (
            pd.DataFrame({
                "gene": var_names,
                "weight": w,  # signed
                "abs_score": abs_score,  # used for ranking
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
            # single-branch child — no direct weight
            # backtrack to nearest ancestor with valid weights
            curr = child
            while curr in child_to_parent:
                parent = child_to_parent[curr]
                if parent in edge_rank_by_child:
                    # inherit parent weights
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


def plot_edge_top_genes_gradient(
    gene_tree,
    edge_id,
    top_n=20,
    figure=(8, 6),
    out_dir=None,
    key="edge_rank_bc",
    cmap_name="RdBu_r",
    dpi=600,
    show=True,
):
    """Plot a signed bar chart of top discriminative genes for an edge.

    Genes are sorted by absolute weight.  Positive weights (up-regulated
    for the child) are shown in red, negative in blue, with the colormap
    centred at zero via ``TwoSlopeNorm``.

    Parameters
    ----------
    gene_tree : dict
        Output of ``rank_multiscale_genes_tree``.
    edge_id : str
        Edge identifier (e.g. ``"root->coarse:1"``).
    top_n : int
        Number of top genes to display.
    figure : tuple
        Figure size (width, height) in inches.
    out_dir : str or None
        If set, save the figure as a PNG to this directory.
    key : str
        Key in gene_tree for edge-level gene rankings.
    cmap_name : str
        Matplotlib diverging colormap name.
    dpi : int
        Output resolution.
    show : bool
        Whether to display the figure.
    """
    if key not in gene_tree:
        raise KeyError(f"gene_tree missing key={key!r}")

    if edge_id not in gene_tree[key]:
        raise KeyError(f"edge_id={edge_id!r} not found")

    df = gene_tree[key][edge_id]

    # 1. sort by abs_score, take top N
    sub = df.sort_values("abs_score", ascending=False).head(int(top_n)).copy()


    # 2. create gradient colour mapping
    # TwoSlopeNorm centers 0 at the colormap midpoint
    v_min, v_max = sub["weight"].min(), sub["weight"].max()
    # prevent all-positive/all-negative from breaking Norm
    if v_min >= 0:
        v_min = -0.0001
    if v_max <= 0:
        v_max = 0.0001
    norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
    cmap = plt.get_cmap(cmap_name)
    color_list = [cmap(norm(x)) for x in sub["weight"]]

    # ---- styling ----
    sns.set_theme(style="white")
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=figure)

    # draw bar chart
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

    # add vertical reference line at 0
    ax.axvline(0, color='black', linewidth=1, zorder=3)

    sns.despine(left=False, bottom=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(f"Discriminative Gene Features: {edge_id}", fontsize=15, weight="bold", pad=25)
    ax.set_xlabel("Weight", fontsize=12, labelpad=10)
    ax.set_ylabel("")


    # style y-axis gene labels
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_style("italic")


    # compute max absolute score for axis limits
    max_abs_score = sub["abs_score"].max()

    # set symmetric x-axis range with buffer for labels
    #
    buffer_factor = 1
    ax.set_xlim(-max_abs_score * buffer_factor, max_abs_score * buffer_factor)


    # get min/max of positive and negative weights
    v_min = sub["weight"].min()
    v_max = sub["weight"].max()

    # set asymmetric buffers for left and right
    # negative range:
    if v_min < 0:
        x_left = v_min * 1.3
    else:
        x_left = -0.0001

    # positive range:
    if v_max > 0:
        x_right = v_max * 1.3
    else:
        x_right = 0.001

    # set asymmetric x-axis limits
    ax.set_xlim(x_left, x_right)

    # re-draw 0 line for visibility with asymmetric limits
    ax.axvline(0, color='black', linewidth=1.2, zorder=3)

    # annotation logic auto-adapts ha based on sign
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

    if show:
        plt.show()
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
    Compute weighted expression score for a given edge.

    Selects top genes via dynamic knee-point detection on absolute weights,
    then computes score = Sum(Pos) - Sum(Neg) across the selected genes.

    Returns
    -------
    np.ndarray
        Per-spot score array, shape (n_obs,).
    dict (optional)
        {pos: list, neg: list, n: int} when return_gene_list=True.
    """

    # 1. get and sort weight data
    if edge_id not in gene_tree[res_key]:
        raise KeyError(f"Edge {edge_id} not found in {res_key}")

    df_edge = gene_tree[res_key][edge_id].copy()
    df_edge = df_edge.sort_values("abs_score", ascending=False).reset_index(drop=True)

    # 2. dynamic knee-point detection for gene count
    scores = df_edge["abs_score"].values
    x = np.arange(len(scores))
    kn = KneeLocator(
        x,
        scores,
        curve='convex',
        direction='decreasing',
        interp_method='interp1d',
        S=Knee_S,
    )
    knee_idx = kn.knee
    if knee_idx is None or knee_idx == 0:
        kn_poly = KneeLocator(
            x,
            scores,
            curve='convex',
            direction='decreasing',
            interp_method='polynomial',
            S=Knee_S,
        )
        knee_idx = kn_poly.knee
    if knee_idx is None or knee_idx < 2:
        knee_idx = min_genes

    dynamic_n = int(np.clip(knee_idx, min_genes, max_genes))

    # 3. extract top genes
    top_genes_df = df_edge.head(dynamic_n)
    pos_genes = [
        g
        for g in top_genes_df[top_genes_df['direction'].str.contains('pos')]['gene']
        if g in adata.var_names
    ]
    neg_genes = [
        g
        for g in top_genes_df[top_genes_df['direction'].str.contains('neg')]['gene']
        if g in adata.var_names
    ]

    # 4. matrix accumulation (Sum(Pos) - Sum(Neg))
    final_scores = np.zeros(adata.n_obs)

    if pos_genes:
        X_pos = adata[:, pos_genes].X
        if hasattr(X_pos, "toarray"):
            X_pos = X_pos.toarray()
        final_scores += X_pos.sum(axis=1).flatten()

    if neg_genes:
        X_neg = adata[:, neg_genes].X
        if hasattr(X_neg, "toarray"):
            X_neg = X_neg.toarray()
        final_scores -= X_neg.sum(axis=1).flatten()

    if return_gene_list:
        return final_scores, {"pos": pos_genes, "neg": neg_genes, "n": dynamic_n}

    return final_scores


def plot_gene_weights_spatial(
    adata,
    gene_tree,
    edge_id,
    res_key="edge_rank_bc",
    out_dir=None,
    filename=None,       # Standard filename parameter
    spot_size=100,
    Knee_S=1,
    show_knee_plot=True,
    cmap='RdYlBu_r',
    figsize=(5, 5),      # Standard figure size
    show=True,
    close=None,          # Standard close control
    dpi=600,             # Standard resolution
    ax=None,             # Use external axes
    base_fontsize=None,  # Base font size; auto-scaled when None
    **kwargs
):
    """
    Spatial visualisation of gene weights for a given edge.
    Features dynamic font scaling and Nature-style formatting compatible with external axes.
    """

    # 1. call calculate_edge_scores
    final_scores, info = calculate_edge_scores(
        adata=adata,
        gene_tree=gene_tree,
        edge_id=edge_id,
        res_key=res_key,
        Knee_S=Knee_S,
        return_gene_list=True
    )

    # 2. prepare safe filename and data column
    safe_edge = edge_id.replace(':', '_').replace('->', '__to__')
    score_name = f"{safe_edge}"
    adata.obs[score_name] = final_scores

    if filename is None and out_dir is not None:
        filename = f"spatial_{safe_edge}.png"

    # 3. Diagnostic knee plot on an independent figure.
    if show_knee_plot or out_dir:
        df_edge = gene_tree[res_key][edge_id].sort_values("abs_score", ascending=False)
        scores_val = df_edge["abs_score"].values

        fig_knee = plt.figure(figsize=(5, 4))
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
            fig_knee.savefig(knee_path, dpi=300, bbox_inches='tight')

        if show_knee_plot:
            plt.show()
        else:
            plt.close(fig_knee)  # Close only the knee plot.

    # ==========================================
    # 4. Dynamic font scaling and global style.
    # ==========================================
    if base_fontsize is None:
        base_fontsize = max(7, min(14, int(figsize[1] * 2.2)))

    nature_rc = {
        "figure.figsize": figsize,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }

    # 5. Main spatial plot rendering.
    with plt.rc_context(nature_rc):
        is_standalone = (ax is None)

        sc.pl.spatial(
            adata,
            color=score_name,
            spot_size=spot_size,
            cmap=cmap,
            show=False,
            ax=ax,         # Attach to the provided axes
            **kwargs
        )

        # Get the current figure and axes.
        if is_standalone:
            fig = plt.gcf()
            main_ax = fig.axes[0] if len(fig.axes) > 0 else None
        else:
            fig = ax.get_figure()
            main_ax = ax

        # 6. Standardized output and saving.
        saved_path = None
        if is_standalone:
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                saved_path = os.path.join(out_dir, filename)
                fig.savefig(saved_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

            if show:
                plt.show()
            elif close is not False:
                plt.close(fig)

    return fig, main_ax, saved_path


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
    """Infer domain/coarse labels from uns maps when missing from obs."""
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
                    raise ValueError(f"fine {f} belongs to both domain {fine_to_domain[f]} and {d}")
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
                    raise ValueError(f"domain {d} belongs to both coarse {domain_to_coarse[d]} and {c}")
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
            # find which coarse this domain belongs to (reverse lookup)
            d = str(d)
            c_of_d = None
            for c, ds in coarse_map.items():
                if d in [str(x) for x in ds]:
                    c_of_d = str(c)
                    break
            if c_of_d is None:
                raise ValueError(f"domain {d} not found in coarse_map")
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
    out_dir=None,
    prefix=None,
):
    """
    Compute CV logistic regression metrics for an edge.

    Optionally saves metrics, genes, proba, and y to disk.
    """
    # ---------- 0) get top-k genes for the edge ----------
    df_edge = None
    edge_full_id = None

    if rank_key in out and edge_id in out[rank_key]:
        df_edge = out[rank_key][edge_id]
        if isinstance(df_edge, list):
            if len(df_edge) == 0:
                raise KeyError(f"out[{rank_key!r}][{edge_id}] is an empty list")
            df_edge = df_edge[0]
        child_id_input = edge_id

    elif "edge_rank" in out and edge_id in out["edge_rank"]:
        df_edge = out["edge_rank"][edge_id]
        edge_full_id = edge_id
        child_id_input = edge_id.split("->", 1)[1]

    else:
        raise KeyError(
            f"edge_id={edge_id} not found in out[{rank_key!r}] or out['edge_rank']."
        )

    genes = df_edge["gene"].head(k).astype(str).tolist()

    # ---------- 1) resolve node_id / child_id (requires task_info) ----------
    if edge_full_id is not None:
        node_id, child_id = edge_full_id.split("->", 1)
    else:
        child_id = str(child_id_input)
        if "task_info" not in out or out["task_info"] is None:
            raise KeyError(
                "out missing task_info; cannot infer node_id from child_id. "
                "Provide the full parent->child edge_id or keep task_info."
            )
        ti = out["task_info"]
        hit = ti[ti["child_id"].astype(str) == child_id]
        if hit.shape[0] == 0:
            raise KeyError(f"child_id={child_id} not found in task_info (edge_id_input={edge_id})")
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
        raise RuntimeError(f"scope has too few samples: {idx.size}")

    # ---------- 4) y ----------
    if child_id.startswith("coarse:"):
        pos_mask = (coarse_series.values == child_id.split(":", 1)[1])
    elif child_id.startswith("domain:"):
        pos_mask = (domain_series.values == child_id.split(":", 1)[1])
    elif child_id.startswith("fine:"):
        pos_mask = (fine_series.values == child_id.split(":", 1)[1])
    else:
        raise ValueError(f"unsupported child_id={child_id}")

    y = pos_mask[idx].astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        raise RuntimeError(f"degenerate edge in scope: pos={y.sum()}, neg={y.size - y.sum()}")

    # ---------- 5) X (top-k genes) ----------
    Xall, var_names = _get_X_and_varnames(adata, use_raw=use_raw, layer=layer)
    g2i = {g: i for i, g in enumerate(var_names)}
    keep_genes = [g for g in genes if g in g2i]
    if len(keep_genes) == 0:
        raise RuntimeError("none of the top-k genes found in var_names (check use_raw/layer or gene names)")

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

        # flatten metrics to one row
        df_metrics = pd.DataFrame([m])
        p_metrics = os.path.join(save_dir, "metrics.csv")
        df_metrics.to_csv(p_metrics, index=False)
        saved["metrics_csv"] = p_metrics

        # 2) confusion_matrix.csv (stored separately)
        p_cm = os.path.join(save_dir, "confusion_matrix.csv")
        pd.DataFrame(cm, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"]).to_csv(p_cm)
        saved["confusion_matrix_csv"] = p_cm

        # 3) genes_used.csv
        p_genes = os.path.join(save_dir, "genes_used.csv")
        pd.DataFrame({"gene": genes_used}).to_csv(p_genes, index=False)
        saved["genes_used_csv"] = p_genes

        # 4) proba_y.csv (y and proba side by side)
        p_py = os.path.join(save_dir, "proba_y.csv")
        pd.DataFrame({"y": y.astype(int), "proba": proba.astype(float)}).to_csv(p_py, index=False)
        saved["proba_y_csv"] = p_py

        # 5) scope_idx.csv (records which samples were used)
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
    out_dir=None,
    prefix=None,
    dpi=600,
    show=True,
):
    """
    Plot ROC / PR / confusion matrix / histogram + spatial.

    Writes a temporary column to obs and removes it afterwards.
    Optionally saves PNGs.
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


    # --------- 1) 2x2 diagnostic plots ---------
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
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
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

    # --------- 2) spatial (temporary column) ---------
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
    Main entry point: compute then plot.
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

    # Stop plotting when computation fails or is skipped.
    if res is None:
        print(f"⚠️ Skipping plotting for '{edge_id}': compute_edge_discrimination returned None.")
        return None

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


def batch_plot_gene_tree_results(
    adata,
    gene_tree,
    base_out_dir,
    spot_size=140,
    top_n=20,
    res_key="edge_rank_bc",
    n_jobs=-1,  # Number of workers; -1 uses all available cores
):
    """
    Batch-process all edges in a gene_tree in parallel.

    Iterates over every edge, generating gradient bar charts, spatial
    plots, and edge discrimination evaluation for multi-task edges.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data.
    gene_tree : dict
        Output of ``rank_multiscale_genes_tree``.
    base_out_dir : str
        Root directory for output (a ``gene_tree/`` subfolder is created).
    spot_size : float
        Spot size for spatial plots.
    top_n : int
        Number of top genes to show in gradient bar charts.
    res_key : str
        Key in gene_tree for edge-level gene rankings.
    n_jobs : int
        Number of parallel jobs to run. -1 means using all processors.
    """
    _adata = adata.copy()

    # 1. create base output directory
    main_dir = os.path.join(base_out_dir, "gene_tree")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Created main directory: {main_dir}")

    # 2. collect all edge_ids
    all_keys = list(gene_tree[res_key].keys())
    print(
        f"Total keys found: {len(all_keys)}, "
        f"utilizing {n_jobs if n_jobs != -1 else 'all available'} CPU cores."
    )

    valid_tasks = set()
    if "task_info" in gene_tree:
        valid_tasks = set(gene_tree["task_info"]["child_id"].unique())

    # ==========================================
    # Extract the loop body into a worker function.
    # ==========================================
    def _process_single_edge(edge_id):
        # sanitize folder name (replace colons)
        safe_key = edge_id.replace(":", "_")
        key_dir = os.path.join(main_dir, safe_key)

        # Use exist_ok=True to avoid parallel directory races.
        os.makedirs(key_dir, exist_ok=True)

        # 3. gradient bar chart
        try:
            # Assume this function is global or defined in this module.
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

        # 4. spatial weight distribution plot
        try:
            plot_gene_weights_spatial(
                _adata,
                gene_tree,
                edge_id=edge_id,
                spot_size=spot_size,
                cmap='RdYlBu_r',
                Knee_S=3,
                show_knee_plot=False,
                show=False,
                out_dir=key_dir
            )
        except Exception as e:
            print(f"Error in spatial plot for {edge_id}: {e}")

        # 5. Discrimination Eval
        is_single_child = edge_id not in valid_tasks
        if not is_single_child:
            try:
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
            except Exception as e:
                print(f"Error in evaluation for {edge_id}: {e}")

    # ==========================================
    # Dispatch tasks with joblib.Parallel.
    # ==========================================
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_edge)(edge_id)
        for edge_id in tqdm(all_keys, desc="Batch Processing Edges")
    )

    print("\nBatch Processing Completed!")
