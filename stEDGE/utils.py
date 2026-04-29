import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from queue import PriorityQueue
import pandas as pd
import os
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading,LabelPropagation
from scipy.cluster.hierarchy import fcluster
import networkx as nx
from collections import defaultdict
from itertools import combinations
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import heapq

def compute_boundary_probability(adata, n_neighbors=6):
    """
    计算每个 spot 的边界概率（局部不一致度），并赋值到 adata.obs["p_edge"]。
    
    参数：
    - adata: AnnData 对象，必须包含 adata.obsm["spatial"]
    - con_clust_mat: 共识聚类矩阵，shape=(n_spots, n_spots)
    - n_neighbors: 用于计算局部邻域的近邻数，默认为6
    """
    coords = adata.obsm["spatial"]
    n_spots = coords.shape[0]
    consensus_freq = adata.obsm["consensus_freq"].copy()
    
    # 用空间坐标构建最近邻索引
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # indices[:, 0] 是每个点自己，所以我们去掉
    neighbor_indices = indices[:, 1:]

    # 计算每个 spot 的不一致度
    inconsistency_scores = []
    for i in range(n_spots):
        neighbor_ids = neighbor_indices[i]
        local_consensus = consensus_freq[i, neighbor_ids]
        local_agreement = np.mean(local_consensus)
        local_inconsistency = 1 - local_agreement  # 越大越像边界
        inconsistency_scores.append(local_inconsistency)

    # 标准化到 [0, 1] 作为边界“概率”
    p_edge = np.array(inconsistency_scores)
    p_edge = (p_edge - p_edge.min()) / (p_edge.max() - p_edge.min() + 1e-8)

    # 存入 adata.obs
    adata.obs["p_edge"] = p_edge
    adata.obs["a_p_edge"] = 1-p_edge

    print(">>> Computed boundary probabilities and stored in adata.obs['p_edge'].")


    return   # 可选，方便链式操作


def compute_gradient(adata, norm = True, k=6):

    p_edge = adata.obs['p_edge'].values.copy()
    coords = adata.obsm['spatial'].copy()

    if norm:
        coords = (coords - coords.mean(axis=0)) / coords.std(axis=0) 
    # 平滑p_edge以减少噪声
    grad = np.zeros_like(coords)
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # 计算平滑后的p_edge
    p_edge_smooth = np.array([p_edge[nb].mean() for nb in indices], dtype=np.float32)

    for i in range(len(coords)):
        neighbors = indices[i]
        # 计算p_edge和坐标的局部差分
        delta_p = p_edge_smooth[neighbors] - p_edge_smooth[i]
        delta_coords = coords[neighbors] - coords[i]
        # 最小二乘拟合梯度
        grad[i] = np.linalg.lstsq(delta_coords, delta_p, rcond=None)[0]
    adata.obsm['grad_p_edge'] = grad
    print(">>> Computed gradient of p_edge and stored in adata.obsm['grad_p_edge'].")
    return 

def find_seed_regions(
    adata,
    p_cap_start=0.20,
    p_cap_step=0.02,
    p_cap_max=0.40,
    n_neighbors=6
):
    coords = adata.obsm["spatial"]
    p_edge = adata.obs["p_edge"].values

    # Step 1: 质量优先的阈值筛选（p_cap 逐步放宽，替换 seed_threshold 的静态逻辑）
    min_seed_frac = 0.1
    min_seed_n = max(1, int(len(p_edge) * min_seed_frac))

    p_cap = p_cap_start
    low_p_indices = np.array([], dtype=int)

    used_fallback = False  

    # 逐步放宽 p_cap，直到 seed 数量达到最低要求（或到达 p_cap_max）
    while p_cap <= p_cap_max + 1e-12:
        seed_mask = p_edge <= p_cap
        low_p_indices = np.where(seed_mask)[0]
        if low_p_indices.size >= min_seed_n:
            break
        p_cap += p_cap_step

    # 到 p_cap_max 仍不够：兜底取最小的 min_seed_n 个（质量最优），不再用 seed_threshold%
    if low_p_indices.size < min_seed_n:
        used_fallback = True
        low_p_indices = np.argsort(p_edge)[:min_seed_n]
        seed_mask = np.zeros_like(p_edge, dtype=bool)
        seed_mask[low_p_indices] = True

    # Step 2: KNN 图构建（只在低边界概率区域之间连边）——保持不动
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    knn_graph = nn.kneighbors(coords, return_distance=False)[:, 1:]

    edges = [
        (i, j)
        for i in low_p_indices
        for j in knn_graph[i]
        if j in low_p_indices
    ]
    G = nx.Graph()
    G.add_edges_from(edges)

    connected_components = list(nx.connected_components(G))

    # Step 3: 生成种子标签 ——保持不动
    labeled_seeds = np.zeros(len(p_edge), dtype=int)
    for seed_label, component in enumerate(connected_components, start=1):
        for idx in component:
            labeled_seeds[idx] = seed_label

    adata.obs["labeled_seeds"] = labeled_seeds
    adata.obsm["knn_graph"] = knn_graph
 
    if used_fallback:
            print(
                f">>> Found {len(connected_components)} seed regions. "
                f"Fallback to smallest {min_seed_n}/{len(p_edge)} points "
                f"(p_cap scanned up to {p_cap_max:.3f}, start={p_cap_start:.3f}, step={p_cap_step:.3f})."
            )
    else:
        print(f">>> Found {len(connected_components)} seed regions with p_cap={p_cap:.3f}.")
    

    return



def region_growing_segmentation(
    adata,
    global_thr_k=0.25,
    output_dir=None,
    save_log=False,
):
    """
    Faster version of region growing segmentation.

    Parameters
    ----------
    adata : AnnData
        Must contain:
        - adata.obsm['spatial']
        - adata.obs['p_edge']
        - adata.obsm['knn_graph']
        - adata.obs['labeled_seeds']
        - adata.obsm['grad_p_edge']
    global_thr_k : float
        Threshold coefficient based on IQR.
    output_dir : str or None
        If provided and save_log=True, save log CSV there.
    save_log : bool
        Whether to store detailed growth log in adata.uns and optionally write CSV.

    Returns
    -------
    None
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    p_edge = np.asarray(adata.obs["p_edge"].values, dtype=np.float64)
    knn_graph = adata.obsm["knn_graph"]
    labeled_seeds = np.asarray(adata.obs["labeled_seeds"].values, dtype=int)
    grad_p_edge = np.asarray(adata.obsm["grad_p_edge"], dtype=np.float64)

    n = len(p_edge)
    labels = np.zeros(n, dtype=np.int32)

    unique_seeds = np.unique(labeled_seeds)
    unique_seeds = unique_seeds[unique_seeds > 0]

    # Global threshold: computed once
    global_q1 = np.percentile(p_edge, 25)
    global_q3 = np.percentile(p_edge, 75)
    global_iqr = global_q3 - global_q1
    global_thr = global_q1 + global_thr_k * global_iqr

    growth_log = [] if save_log else None

    for seed_label in unique_seeds:
        seed_mask = labeled_seeds == seed_label
        seed_indices = np.where(seed_mask)[0]

        if seed_indices.size == 0:
            continue

        # Skip seeds already labeled by previous growth
        unlabeled_seed_indices = seed_indices[labels[seed_indices] == 0]
        if unlabeled_seed_indices.size == 0:
            continue

        # Assign seed labels immediately
        labels[unlabeled_seed_indices] = seed_label

        region_indices = set(unlabeled_seed_indices.tolist())
        region_pe = p_edge[unlabeled_seed_indices].tolist()

        # Compute initial local threshold once
        if len(region_pe) > 0:
            local_q1 = np.percentile(region_pe, 25)
            local_q3 = np.percentile(region_pe, 75)
            local_iqr = local_q3 - local_q1
            upper_bound = local_q3 + global_thr_k * local_iqr
        else:
            upper_bound = global_thr

        # Min-heap queue
        heap = []
        in_queue = np.zeros(n, dtype=bool)

        # Initialize frontier from seeds
        for i in unlabeled_seed_indices:
            neighbors = knn_graph[i]
            for j in neighbors:
                j = int(j)
                if labels[j] > 0 or in_queue[j]:
                    continue

                direction = coords[j] - coords[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction /= norm

                priority = -np.dot(grad_p_edge[i], direction) / (p_edge[j] + 1e-8)
                heapq.heappush(heap, (priority, j))
                in_queue[j] = True

        # Region growing
        while heap:
            _, j = heapq.heappop(heap)
            in_queue[j] = False

            if labels[j] > 0:
                continue

            pe_j = p_edge[j]
            current_thr = max(global_thr, upper_bound)

            if save_log:
                growth_log.append(
                    {
                        "seed_label": int(seed_label),
                        "current_step_node": int(j),
                        "region_size": int(len(region_pe)),
                        "upper_bound": float(upper_bound),
                        "global_thr": float(global_thr),
                        "current_thr": float(current_thr),
                        "pe_j": float(pe_j),
                        "accepted": bool(pe_j <= current_thr),
                    }
                )

            if pe_j > current_thr:
                continue

            # Accept node
            labels[j] = seed_label
            region_indices.add(j)
            region_pe.append(pe_j)

            # Update local threshold ONLY after region grows
            local_q1 = np.percentile(region_pe, 25)
            local_q3 = np.percentile(region_pe, 75)
            local_iqr = local_q3 - local_q1
            upper_bound = local_q3 + global_thr_k * local_iqr

            # Push neighbors
            neighbors = knn_graph[j]
            for k in neighbors:
                k = int(k)
                if labels[k] > 0 or in_queue[k]:
                    continue

                direction = coords[k] - coords[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction /= norm

                priority = -np.dot(grad_p_edge[j], direction) / (p_edge[k] + 1e-8)
                heapq.heappush(heap, (priority, k))
                in_queue[k] = True

    # Write back to AnnData
    adata.obs["region_label"] = pd.Categorical(labels)

    if save_log:
        log_df = pd.DataFrame(growth_log)
        adata.uns["region_growing_log"] = log_df

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "region_growing_log.csv")
            log_df.to_csv(log_path, index=False)
            print(f"    - Log data saved to {log_path}.")
    else:
        adata.uns["region_growing_log"] = pd.DataFrame()

    print(">>> Completed region growing segmentation.")
    print("    - Segmentation labels stored in adata.obs['region_label'].")
    print("    - Log data stored in adata.uns['region_growing_log'].")

    return



def propagate_domains_from_regions(adata, con_th=0.7, detect_unassigned=False, con_th_inner=0.7, min_size=20):
    """
    从 region_label 出发，基于共识频率加权平均进行大簇扩散。支持逐步可视化保存。

    参数：
    - adata: AnnData 对象，要求有 .obs["region_label"] 和 .obsm["consensus_freq"]
    - con_th: 加权平均共识频率阈值
    返回：
    - con_region: 最终大簇标签，写入 adata.obs["con_region"]
    """

    region_labels = adata.obs["region_label"].values
    consensus_freq = adata.obsm["consensus_freq"]
    n_spots = consensus_freq.shape[0]

    con_region = np.zeros(n_spots, dtype=int)
    assigned = set()
    current_label = 1

    # 簇按大小排序
    region_ids, region_sizes = np.unique(region_labels, return_counts=True)
    region_info = [(r, s) for r, s in zip(region_ids, region_sizes) if r != 0]
    region_info.sort(key=lambda x: -x[1])
    sorted_regions = [r for r, _ in region_info]


    coords = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=10).fit(coords)

    for r in sorted_regions:
        region_indices = np.where(region_labels == r)[0]
        if all(idx in assigned for idx in region_indices):
            continue

        # 计算加权平均
        cluster_scores = np.zeros(n_spots)
        for idx in region_indices:
            cluster_scores += consensus_freq[idx]
        cluster_scores /= len(region_indices)

        cluster_indices = np.where(cluster_scores >= con_th)[0]
        cluster_indices = [idx for idx in cluster_indices if idx not in assigned]
        if not cluster_indices:
            continue

        # 分配标签前：扩展 cluster_indices，加入完全被包围的点
        # 当前小簇已识别点，不应参与包围检查
        unassigned_indices = np.array([i for i in range(n_spots) 
                                       if i not in assigned and i not in cluster_indices])
        cluster_set = set(cluster_indices)
        internal_candidates = []
        distances, neighbors = nbrs.kneighbors(coords[unassigned_indices])
        neighbors = neighbors[:, 1:] #去除自身

        for i, neigh in enumerate(neighbors):
            global_idx = unassigned_indices[i]
            if all(n in cluster_set for n in neigh):
                internal_candidates.append(global_idx)

        # 合并扩展点
        cluster_indices = list(cluster_set.union(internal_candidates))

        # ➕ 剔除孤立点
        cluster_set = set(cluster_indices)
        isolated_candidates = []
        distances_iso, neighbors_iso = nbrs.kneighbors(coords[cluster_indices])
        neighbors_iso = neighbors_iso[:, 1:]
        
        for i, neigh in enumerate(neighbors_iso):
            global_idx = cluster_indices[i]
            if all(n not in cluster_set for n in neigh):
                isolated_candidates.append(global_idx)
        
        cluster_indices = [i for i in cluster_indices if i not in isolated_candidates]
        

        # 分配标签
        for idx in cluster_indices:
            con_region[idx] = current_label
        assigned.update(cluster_indices)
        current_label += 1

    adata.obs["con_region"] = pd.Categorical(con_region)
    print(">>> Completed domain propagation by Con.")
    print("    - Con domain labels stored in adata.obs['con_region'].")

    if detect_unassigned:
        print(">>> Detected new domains from unassigned points.")

        detect_new_domains_from_unassigned(
            adata,
            con_th_inner=con_th_inner,
            min_size=min_size,
            con_region_key="con_region"
        )

    return 

def detect_new_domains_from_unassigned(
    adata,
    con_th_inner=0.7,
    min_size=20,
    con_region_key="con_region"
):
    """
    在 con_region=0 的未分配点中，基于共识频率寻找内部一致的子群体，生成新域。

    参数:
    - adata: AnnData 对象，要求包含 obsm['consensus_freq'] 和 obs[con_region_key]
    - th_inner: float, 两个点连边的共识频率阈值
    - min_size: int, 连通分量最小点数，小于该值的忽略
    - con_region_key: str, 在 obs 中的列名
    
    返回:
    - 更新后的 adata.obs[con_region_key]（Categorical）
    """
    con_region = adata.obs[con_region_key]
    # 找出未分配点
    unassigned = np.where(con_region.astype(int).values == 0)[0]
    total_unassigned = len(unassigned)
    if len(unassigned) == 0:
        print("No unassigned points to process.")
        return adata.obs["con_region"]

    # 子矩阵
    sub_freq = adata.obsm['consensus_freq'][unassigned][:, unassigned]

    # 构建邻接矩阵
    adj_matrix = (sub_freq >= con_th_inner).astype(int)

    # 构建图
    G = nx.from_numpy_array(adj_matrix)

    # 找到连通分量
    components = list(nx.connected_components(G))

    # 当前最大标签
    max_label = con_region.astype(int).max()
    new_domains_count = 0
    discarded_small = 0
    domain_sizes = []

    for comp in components:
        if len(comp) < min_size:
            discarded_small += 1
            continue
        max_label += 1
        new_domains_count += 1
        domain_sizes.append(len(comp))
        con_region = con_region.cat.add_categories([max_label])
        for idx in comp:
            global_idx = unassigned[idx]
            con_region.iloc[global_idx] = max_label

    # 写回 AnnData
    adata.obs["con_region"] = con_region

    # info
    info = {
        "total_unassigned": total_unassigned,
        "new_domains_count": new_domains_count,
        "domain_sizes": domain_sizes,
        "discarded_small_components": discarded_small,
        "final_max_label": max_label
    }
    adata.uns["new_unassigned_domain_info"] = info
    print(f"  -New domains detected from unassigned points: {new_domains_count}")

    return 

def label_spreading_with_pca(
    adata,
    n_components=50,
    n_neighbors=7,
    alpha=0.8,
):
    """
    使用 PCA 降维后的表达特征进行 LabelSpreading，补全未标记的标签。
    
    参数：
    - adata: AnnData 对象，adata.X 为原始特征
    - label_key: 部分标签列名（未标记为 0）
    - output_key: 存储预测标签的 obs 列名
    - n_components: PCA 降维维度
    - alpha: LabelSpreading 平滑参数
    - gamma: RBF 核的参数
    """
    # Step 1: PCA 降维
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(adata.X)

    # Step 2: 标签准备
    labels = adata.obs["con_region"].astype(int).values
    y = np.where(labels == 0, -1, labels)  # sklearn 格式要求

    # Step 3: 标签传播
    model = LabelSpreading(
    kernel='knn',
    n_neighbors=n_neighbors,
    alpha=alpha,
    max_iter=1000
        )
    model.fit(X_pca, y)

    # Step 4: 存储结果
    adata.obs["fine"] = model.transduction_

    return model

def label_propagation_with_pca(
    adata,
    n_components=50,
    n_neighbors=7,
):
    """
    使用 PCA 降维后的表达特征进行 LabelPropagation，补全未标记的标签，
    并保证原始已标注的标签不会被覆盖。

    参数：
    - adata: AnnData 对象，adata.X 为原始特征
    - n_components: PCA 降维维度
    - n_neighbors: kNN 邻居数
    - max_iter: 最大迭代次数
    - label_key: 输入的标签列名（未标记为 0）
    - output_key: 输出预测标签的列名
    """

    # Step 1: PCA 降维
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(adata.X)

    # Step 2: 标签准备
    orig_labels = adata.obs["con_region"].astype(int).values
    y = np.where(orig_labels == 0, -1, orig_labels)  # -1 表示未知标签

    # Step 3: 标签传播
    model = LabelPropagation(
        kernel='knn',
        n_neighbors=n_neighbors,
        max_iter=1000
    )
    model.fit(X_pca, y)

    # Step 4: 存储结果（保护原始已标注点）
    new_labels = model.transduction_.copy()
    mask = orig_labels != 0
    new_labels[mask] = orig_labels[mask]  # 锁定已知标签

    adata.obs["fine"] = new_labels.astype(int)
    adata.obs["fine"] = adata.obs["fine"].astype("category")
    print(">>> Completed domain propagation with LabelPropagation.")
    print("    - Propagated labels stored in adata.obs['fine'].")

    return model

def merge_clusters_by_dendrogram(
    adata,
    groupby='fine',
    max_clusters=None,
    distance_threshold=None,
    cor_method='pearson',
    linkage_method='ward',
    dendrogram_key=None,
    label_key_added='domain',
    min_size= 10
):
    """
    基于树状图合并已有的分类标签（如 domain）。

    参数:
    - adata: AnnData
        输入 AnnData 对象。
    - groupby: str
        用于聚类的列（如 adata.obs['fine']）。
    - max_clusters: int | None
        指定最终保留的簇数。如果设置了，则优先使用。
    - distance_threshold: float | None
        根据树状图距离进行分割。
    - cor_method: str
        相关性方法（如 'pearson'）。
    - linkage_method: str
        层次聚类方式（如 'ward', 'average'）。
    - dendrogram_key: str | None
        保存到 adata.uns 的键名。默认根据 groupby 自动生成。
    - label_key_added: str
        合并结果写入 adata.obs 的列名。

    返回:
    - merged_labels: pd.Categorical
        合并后的标签分类。
    """

    if dendrogram_key is None:
        dendrogram_key = f'dendrogram_{groupby}'


    # Step 1: 构建 dendrogram，得到 correlation matrix
    sc.tl.dendrogram(adata, groupby=groupby, cor_method=cor_method,
                     linkage_method=linkage_method, key_added=dendrogram_key, inplace=True)

    adata.uns[f"dendrogram_{groupby}_ori"] = adata.uns[dendrogram_key].copy()
    adata.obs[f"{groupby}_ori"] = adata.obs[groupby].copy()

    # ---------- Step 0: 小簇预合并 ----------
    group_sizes = adata.obs[groupby].value_counts()
    small_regions = group_sizes[group_sizes < min_size].index.tolist()
    if len(small_regions) > 0:
        merge_small_clusters_by_similarity(
                adata,
                groupby='fine',
                min_size=min_size,
                merged_label_key="fine",
                dendrogram_key=None,
            )
        del adata.uns[dendrogram_key]  # 清理 dendrogram 数据
        sc.tl.dendrogram(adata, groupby=groupby, cor_method=cor_method,
                    linkage_method=linkage_method, key_added=dendrogram_key, inplace=True)

    # Step 2: 获取树状图结构
    Z = adata.uns[dendrogram_key]['linkage']

    # Step 3: 聚类合并
    if max_clusters is not None:
        merge_labels = fcluster(Z, t=max_clusters, criterion='maxclust')
    elif distance_threshold is not None:
        merge_labels = fcluster(Z, t=distance_threshold, criterion='distance')
    else:
        raise ValueError("必须指定 max_clusters 或 distance_threshold 其中之一。")

    # Step 4: 建立映射
    region_labels = adata.obs[groupby]
    valid_regions = region_labels.cat.categories if hasattr(region_labels, 'cat') else np.unique(region_labels)
    region_to_merge = {region: merge_label for region, merge_label in zip(valid_regions, merge_labels)}

    # Step 5: 映射回 obs
    merged_spot_labels = [region_to_merge.get(r, 0) if r != 0 else 0 for r in region_labels]
    adata.obs[label_key_added] = pd.Categorical(merged_spot_labels)

    # Step 6: 还原原始标签
    adata.obs[f"{groupby}"] = adata.obs[f"{groupby}_ori"].copy()
    adata.obs[f"{groupby}"]  = adata.obs[f"{groupby}"].astype("category")
    del adata.obs[f"{groupby}_ori"]

    return adata.obs[label_key_added]


def merge_small_clusters_by_similarity(
    adata,
    groupby='fine',
    min_size=10,
    merged_label_key=None,
    dendrogram_key=None,
):
    """
    将小簇合并到相关性最高的大簇（通过树状图的相关矩阵）。

    参数：
    - adata: AnnData 对象
    - groupby: str，分组列名
    - min_size: int，小簇的最小 size 阈值
    - cor_method: str，相关性方法
    - linkage_method: str，层次聚类方法
    - merged_label_key: str，合并后标签写入 adata.obs 的列名，默认加 _merged
    - dendrogram_key: str，dendrogram 保存键名，默认自动生成

    返回：
    - pd.Categorical，新的合并标签
    """

    if merged_label_key is None:
        merged_label_key = f"{groupby}"
    if dendrogram_key is None:
        dendrogram_key = f"dendrogram_{groupby}"


    # Step 1: 统计小簇/大簇
    group_sizes = adata.obs[groupby].value_counts()
    small_regions = group_sizes[group_sizes < min_size].index.tolist()
    large_regions = group_sizes[group_sizes >= min_size].index.tolist()

    # Step 2: region -> index
    cat_list = list(adata.obs[groupby].cat.categories)
    corr_array = adata.uns[dendrogram_key]["correlation_matrix"]
    corr_df = pd.DataFrame(corr_array, index=cat_list, columns=cat_list)

    # Step 3: 构建 merge 映射：小簇 -> 最相似大簇
    merge_map = {}
    for small in small_regions:
        similarities = corr_df.loc[small, large_regions]
        best_match = similarities.idxmax()
        merge_map[small] = best_match

    # Step 4: 替换小簇标签
    new_labels = adata.obs[groupby].astype(int).copy()
    for small_region, large_region in merge_map.items():
        new_labels[new_labels == small_region] = large_region

    adata.obs[merged_label_key] = pd.Categorical(new_labels)
    adata.obs[merged_label_key] = adata.obs[merged_label_key].astype("category")

    for s, l in merge_map.items():
        print(f"  {s} -> {l}")

    return adata.obs[merged_label_key]



###########################DTI###########################

def compute_transition_scores_entropy(
    adata,
    cons_key: str = "consensus_freq",
    domain_key: str = "domain",
    eps: float = 1e-12,
):
    """
    基于共识矩阵和最终域标签，计算“信息熵型”的：
      1. 每个 spot 的过渡分数 TS_entropy  (0 ~ 1)
      2. 每个 domain 的 DTI_entropy（Domain Transition Index, entropy 版）

    设计思想：
      - 对每个 spot i，有软归属概率 P(i,d)，d=1..K
      - 点级过渡度 TS_entropy(i) = H(P(i,·)) / log(K)，其中 H 为 Shannon 熵：
            H(i) = -sum_d P(i,d) log P(i,d)
        归一化后 TS_entropy(i) ∈ [0,1]：
          · 接近 0：soft 概率接近 one-hot，几乎确定属于某域（非过渡）
          · 接近 1：soft 概率接近均匀，多域高度混合（强过渡 / 不确定）
      - 域级 DTI_entropy(d) = 域内 TS_entropy(i) 的平均值

    参数
    ----
    adata : AnnData
        需要包含：
          - adata.obsm[cons_key]  : 共识矩阵 (N, N)
          - adata.obs[domain_key] : 最终域标签
    cons_key : str
        共识矩阵在 obsm 中的 key，默认 "consensus_freq"
    domain_key : str
        域标签在 obs 中的列名，默认 "domain"
    eps : float
        防止 log(0) 的下界截断

    返回
    ----
    df_dti : pd.DataFrame
        index = domain（字符串），包含一列 DTI_entropy
    """

    # ---------- 0. 取出共识矩阵和标签 ----------
    C = adata.obsm[cons_key]
    if not isinstance(C, np.ndarray):
        C = np.asarray(C)

    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "consensus_freq 必须是 N×N 矩阵"

    # domain 标签，统一转成字符串
    domains_raw = adata.obs[domain_key]
    domains_str = domains_raw.astype(str).to_numpy()

    # ---------- 1. 为每个 domain 建 index 列表 ----------
    unique_domains = sorted(set(domains_str))

    domain_to_idx = {
        d: np.where(domains_str == d)[0] for d in unique_domains
    }

    # ---------- 2. 计算 S(i,d)：共识空间里对每个 domain 的亲和度 ----------
    K = len(unique_domains)
    S = np.zeros((N, K), dtype=float)

    for k, d in enumerate(unique_domains):
        idx_d = domain_to_idx[d]
        if len(idx_d) == 0:
            continue
        # 对每个 i：与 domain d 内所有点的平均共识
        # C[:, idx_d] 形状 (N, |d|)
        S[:, k] = C[:, idx_d].mean(axis=1)

    # ---------- 3. 归一化成 P(i,d)：软归属概率 ----------
    row_sum = S.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0  # 防止除零
    P = S / row_sum  # 形状 (N, K)

    # ---------- 4. 信息熵型 TS：TS_entropy(i) = H(P(i,·)) / log(K) ----------
    # 防止 log(0)
    P_clamped = np.clip(P, eps, 1.0)
    logP = np.log(P_clamped)
    H = -np.sum(P_clamped * logP, axis=1)  # 长度 N

    if K > 1:
        TS_entropy = H / np.log(K)   # 归一化到 [0,1]
        TS_entropy = np.clip(TS_entropy, 0.0, 1.0)
    else:
        TS_entropy = np.zeros_like(H)

    # 写回 obs
    adata.obs[f"{domain_key}_TS"] = TS_entropy

    # ---------- 5. 计算每个 domain 的 DTI_entropy(d) = mean TS_entropy(i) ----------
    rows = []
    domain_DTI_map = {}

    for d in unique_domains:
        idx_d = domain_to_idx[d]
        if len(idx_d) == 0:
            continue
        dti = TS_entropy[idx_d].mean()
        rows.append([d, dti])
        domain_DTI_map[d] = dti

    df_dti = pd.DataFrame(rows, columns=[f"{domain_key}_domain", f"{domain_key}_DTI"]).set_index(f"{domain_key}_domain")

    # ---------- 6. 把 DTI_entropy 映射回每个 spot ----------
    domain_DTI_list = []
    for d_str in domains_str:
        if d_str in domain_DTI_map:
            domain_DTI_list.append(domain_DTI_map[d_str])
        else:
            domain_DTI_list.append(np.nan)

    adata.obs[f"{domain_key}_DTI"] = np.array(domain_DTI_list, dtype=float)

    # ---------- 7. 把 S / P 也存起来，方便后续分析 ----------
    # S: 共识亲和度（未归一化），P: 软概率
    # P 的列顺序对应 unique_domains 的顺序
    adata.obsm[f"{domain_key}_soft_affinity"] = S
    adata.obsm[f"{domain_key}_soft_prob"] = P
    adata.uns[f"{domain_key}_soft_domains"] = np.array(unique_domains, dtype=object)
    adata.uns[f"{domain_key}_DTI"] = df_dti
    # print(f">>> Computed DTI for '{domain_key}' and stored in adata.obs and adata.uns.")
    # print(f"    - Spot-level TS stored in adata.obs['{domain_key}_TS'].")
    # print(f"    - Spot-level DTI stored in adata.obs['{domain_key}_DTI'].")
    # print(f"    - Domain-level DTI stored in adata.uns['{domain_key}_DTI'].")
    # print(f"    - Soft probabilities stored in adata.obsm['{domain_key}_soft_prob'].")
    # print(f"    - Soft domains list stored in adata.uns['{domain_key}_soft_domains'].")
    # print(f"    - Soft affinities stored in adata.obsm['{domain_key}_soft_affinity'].")

    return 


def compute_domain_sim(
    adata,
    domain_key: str = "domain",

):
    """
    基于 adata.obsm['domain_soft_prob'] 构造 domain 之间的相似度 / 距离矩阵。

    相似度定义为：
        sim(a,b) = mean_i [ P(i,a) * P(i,b) ]   （ spots 上的共同归属度）

    然后归一化到 [0,1]，并定义距离：
        dist(a,b) = 1 - sim_norm(a,b)

    结果写入：
      - adata.uns[similarity_key] : pd.DataFrame, index/columns = domain 名称
      - adata.uns[distance_key]   : 同上
    """

    prob_key=f"{domain_key}_soft_prob"
    soft_domains_key=f"{domain_key}_soft_domains"
    distance_key=f"{domain_key}_distance"
    similarity_key=f"{domain_key}_similarity"

    P = adata.obsm[prob_key]                    # 形状 (N, K)
    domains = np.array(adata.uns[soft_domains_key], dtype=str)  # 长度 K

    # 共现矩阵：sum_i P(i,a)*P(i,b)
    S = P.T @ P    # 形状 (K, K)
    N = P.shape[0]
    S = S / N      # 变成 mean_i[P(i,a)P(i,b)]

    # 可选：去掉对角线（自己和自己没必要关心）
    np.fill_diagonal(S, 0.0)

    # 归一化到 [0,1] 之间
    upper = S[np.triu_indices_from(S, k=1)]
    max_val = upper.max() if upper.size > 0 else 1.0
    if max_val == 0:
        S_norm = S
    else:
        S_norm = S / max_val

    # 相似度矩阵（0~1，越大越相似）
    sim_df = pd.DataFrame(S_norm, index=domains, columns=domains)

    # 距离矩阵（0~1，越大越远）
    dist_df = 1.0 - sim_df

    adata.uns[similarity_key] = sim_df
    adata.uns[distance_key] = dist_df
    # print(f">>> Computed domain similarity and distance for '{domain_key}'.")
    # print(f"    - Similarity matrix stored in adata.uns['{similarity_key}'].")
    # print(f"    - Distance matrix stored in adata.uns['{distance_key}'].")
    
    return 


def compute_domain_adjacency(
    adata,
    domain_key="domain",
    n_neighbors=6,
    uns_key=None 
):
    """
    计算 domain 之间的邻接关系及边界强度，并存储到 adata.uns 中。
    """
    coords = adata.obsm["spatial"]
    dom = adata.obs[domain_key].to_numpy()
    n = coords.shape[0]

    # 1. 计算 KNN
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_indices = indices[:, 1:]

    adj_pairs = set()
    boundary_pairs = defaultdict(list)

    p_edge = adata.obs["p_edge"].to_numpy() if "p_edge" in adata.obs.columns else None

    # 2. 遍历 Spot 寻找跨 domain 的边
    for i in range(n):
        di = dom[i]
        for j in neighbor_indices[i]:
            dj = dom[j]
            if di == dj:
                continue
            
            # 排序确保无向性 (d1, d2)
            a, b = sorted([str(di), str(dj)]) # 统一转为字符串处理
            pair = (a, b)
            adj_pairs.add(pair)
            
            if p_edge is not None:
                # 记录该接触对上的边界概率均值
                boundary_pairs[pair].append((p_edge[i] + p_edge[j]) / 2)

    adj_list = list(adj_pairs)

    # 3. 计算边界强度并格式化
    boundary_strength_data = []
    if p_edge is not None:
        for (d1, d2), vals in boundary_pairs.items():
            boundary_strength_data.append({
                "domain_a": d1,
                "domain_b": d2,
                "strength": float(np.mean(vals))
            })
    
    # 4. 存储到 adata.uns
    if uns_key is None:
        uns_key = f"{domain_key}_boundary_strength"
    
    # 转换为 DataFrame 存储更方便后续查询和画图
    df_strength = pd.DataFrame(boundary_strength_data)
    adata.uns[uns_key] = df_strength
    
    print(f">>> Adjacency pairs calculated. Boundary strength stored in adata.uns['{uns_key}'].")

    
    return  



################################ merge_clusters_by_DTI ###########################

# ---- 辅助函数：cluster 间相似度（基于 base sim_df） ----
def _cluster_similarity(c1, c2, cluster_members, sim_df, base_size=None):
    """
    计算两个 cluster 之间的相似度：
      sim(cluster1, cluster2) = size 加权的 base domains 相似度平均
    """
    members1 = cluster_members[c1]
    members2 = cluster_members[c2]

    num = 0.0
    den = 0.0
    for d1 in members1:
        for d2 in members2:
            if d1 not in sim_df.index or d2 not in sim_df.columns:
                continue
            s = sim_df.loc[d1, d2]
            if base_size is not None:
                w = base_size.get(d1, 1.0) * base_size.get(d2, 1.0)
            else:
                w = 1.0
            num += s * w
            den += w
    if den == 0:
        return 0.0
    return num / den


# ---- 辅助函数：cluster 间边界强度（基于原始 boundary_strength） ----
def _cluster_boundary_strength(c1, c2, cluster_members, boundary_strength):
    """
    cluster 间的边界强度 = 所有跨 cluster 的 domain 对的 boundary_strength 平均
    若没有边界信息，则返回 0
    """
    members1 = cluster_members[c1]
    members2 = cluster_members[c2]

    vals = []
    for d1 in members1:
        for d2 in members2:
            a, b = sorted([d1, d2])
            key = (a, b)
            if key in boundary_strength:
                vals.append(boundary_strength[key])
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))



def merge_clusters_by_DTI(
    adata,
    domain_key: str = "fine",
    new_domain_key: str = "domain",
    target_n_domains: int = None,
    min_gain: float = 0.0,
    w_sim: float = 1.0,
    w_dti: float = 0.5,
    w_boundary: float = 0.5,
):
    # -------- 0. 统一 domain label 为字符串 --------
    adata.obs[domain_key] = adata.obs[domain_key].astype(str)



    # 相似度矩阵：按你现在的约定从 uns 读
    sim_df = adata.uns[f"{domain_key}_similarity"].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)

    domain_dti_key = f"{domain_key}_DTI"

    base_domains = sorted(set(adata.obs[domain_key].unique()) & set(sim_df.index))

    counts = adata.obs[domain_key].value_counts()
    base_size = {d: float(counts.get(d, 0.0)) for d in base_domains}

    if domain_dti_key in adata.obs.columns:
        df_dti = (
            adata.obs[[domain_key, domain_dti_key]]
            .dropna()
            .groupby(domain_key)[domain_dti_key]
            .mean()
        )
        base_dti = df_dti.to_dict()
    else:
        base_dti = {d: 0.0 for d in base_domains}

    # -------- 1. 初始 cluster --------
    cluster_members = {d: {d} for d in base_domains}
    cluster_dti = {d: base_dti.get(d, np.nan) for d in base_domains}
    domain_to_cluster = {d: d for d in base_domains}

    # 边界字典 key 统一成 (min,max)
    boundary_strength = adata.uns.get(f"{domain_key}_boundary_strength", pd.DataFrame()).set_index(
        ["domain_a", "domain_b"]
    )["strength"].to_dict()

    gain_log = [] 

    base_labels = adata.obs[domain_key].astype(str).to_numpy()
    # **初始整体熵（按初始划分）**
    cur_labels = np.array([domain_to_cluster.get(d, d) for d in base_labels])
    global_entropy, _ = compute_partition_entropy_from_labels(
        adata,
        labels=cur_labels,
        cons_key="consensus_freq",
        normalized=True,
    )

    # -------- 2. 迭代合并（全局）--------
    merge_id = 0
    step = 0
    while True:
        clusters = list(cluster_members.keys())
        n_clusters = len(clusters)

        if target_n_domains is not None and n_clusters <= target_n_domains:
            break
        if n_clusters < 2:
            break

        best_gain = -np.inf
        best_pair = None

        for (c1, c2) in combinations(clusters, 2):
            sim_val = _cluster_similarity(c1, c2, cluster_members, sim_df, base_size)

            dti1 = cluster_dti.get(c1, np.nan)
            dti2 = cluster_dti.get(c2, np.nan)
            if np.isnan(dti1) or np.isnan(dti2):
                dti_term = 0.0
            else:
                dti_term = 1.0 - max(dti1, dti2)

            b_val = _cluster_boundary_strength(c1, c2, cluster_members, boundary_strength)

            gain = w_sim * sim_val + w_dti * dti_term - w_boundary * b_val

            if gain > best_gain:
                best_gain = gain
                best_pair = (c1, c2)

        if best_pair is None or best_gain < min_gain:
            break

        c1, c2 = best_pair
        merge_id += 1
        new_c = f"C{merge_id:04d}"

        members_new = cluster_members[c1] | cluster_members[c2]
        cluster_members[new_c] = members_new


        # cluster DTI：base_size 加权平均
        dti_vals, weights = [], []
        for d in members_new:
            dti_d = base_dti.get(d, np.nan)
            if not np.isnan(dti_d):
                w = base_size.get(d, 1.0)
                dti_vals.append(dti_d)
                weights.append(w)
        cluster_dti[new_c] = np.nan if len(dti_vals) == 0 else float(np.average(dti_vals, weights=weights))

        del cluster_members[c1]; del cluster_members[c2]
        cluster_dti.pop(c1, None); cluster_dti.pop(c2, None)

        for d in members_new:
            domain_to_cluster[d] = new_c

        # **合并之后，重新计算整体熵**  
        step += 1
        cur_labels = np.array([domain_to_cluster.get(d, d) for d in base_labels])
        new_entropy, _ = compute_partition_entropy_from_labels(
            adata,
            labels=cur_labels,
            cons_key="consensus_freq",
            normalized=True,
        )
        gain_log.append({
            "step": step,
            "n_clusters_before": n_clusters,
            "n_clusters_after": len(cluster_members),
            "best_gain": float(best_gain),
            "best_pair": f"{c1}__{c2}",
            "entropy_before": float(global_entropy),
            "entropy_after": float(new_entropy),
            "entropy_delta": float(new_entropy - global_entropy),
        })
        global_entropy = new_entropy

        

    # -------- 3. 统一重命名为 1, 2, ... 并写回 --------
    
    # 3.1 给最终 cluster 生成稳定的 1/2/...（这里按 cluster size 从大到小排序，你也可换成别的排序规则）
    final_clusters = list(cluster_members.keys())
    final_clusters_sorted = sorted(
        final_clusters,
        key=lambda c: sum(base_size.get(d, 1.0) for d in cluster_members[c]),
        reverse=True
    )
    
    cluster_rename = {c: f"{i+1}" for i, c in enumerate(final_clusters_sorted)}
    
    meta_map = {
        cluster_rename[c]: sorted(list(cluster_members[c]))
        for c in final_clusters_sorted
    }
    
    # 3.3 写回 obs：先映射到临时 cluster，再映射到 M*
    old_labels = adata.obs[domain_key].astype(str).tolist()
    new_labels_tmp = [domain_to_cluster.get(d, d) for d in old_labels]  # 临时 cluster 名或原值
    new_labels_final = [cluster_rename.get(x, x) for x in new_labels_tmp]  # 临时名 -> M*，未参与合并的保持原值
    
    adata.obs[new_domain_key] = pd.Categorical(new_labels_final)
    adata.obs[new_domain_key] = adata.obs[new_domain_key].astype('int')
    adata.obs[new_domain_key] = adata.obs[new_domain_key].astype('category')
    
    # 3.4 保存映射关系
    adata.uns[new_domain_key + "_map"] = meta_map
    adata.uns[new_domain_key + "_gain_log"] = pd.DataFrame(gain_log)

    df_dti = compute_transition_scores_entropy(
            adata,
            domain_key = new_domain_key,
        )

    compute_domain_sim(
            adata,
            domain_key = new_domain_key,
        )
    
    # print(f">>> Merged clusters into '{new_domain_key}' with {len(meta_map)} domains.")
    # print(f"    - Mappings stored in adata.uns['{new_domain_key}_map'].")
    # print(f"    - Gain log stored in adata.uns['{new_domain_key}_gain_log'].")

    return meta_map



def search_coarse_domains(
    adata,
    domain_key: str = "domain",
    min_sim_th: float = 0.1,
    resolution: float = 1.0,
):
    """
    在 domain graph 上对域点做聚类（社区检测），得到“成簇的域点”。

    - 节点：domain（来自 adata.obs[domain_key]）
    - 边权：domain 相似度（来自 adata.uns[similarity_key]）
    - 聚类算法：Leiden（scanpy.tl.leiden，在域级图上运行）

    输出：
    - adata.obs[new_cluster_key] : 每个 spot 对应的 domain_cluster id（category）
    - adata.uns[new_cluster_key + "_map"] : {cluster_id: [domain1, domain2, ...]}
    """
    new_cluster_key= f"{domain_key}_coarse"

    # ---- 1. 取相似度矩阵 ----

    similarity_key = f"{domain_key}_similarity"

    if similarity_key not in adata.uns:
        raise KeyError(
            f"{similarity_key} 不在 adata.uns 中，请先调用 compute_domain_graph_from_soft_prob "
            f"为 {domain_key} 构建相似度矩阵。"
        )

    sim_df = adata.uns[similarity_key].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)

    domains = sim_df.index.to_list()
    K = len(domains)
    if K == 0:
        raise ValueError("相似度矩阵为空，没有可聚类的 domain。")

    # ---- 2. 构建加权邻接矩阵（域级图）----
    sim = sim_df.to_numpy()
    # 去掉对角线
    np.fill_diagonal(sim, 0.0)

    # 阈值过滤掉太小的相似度，避免噪声边
    if min_sim_th > 0:
        sim = np.where(sim >= min_sim_th, sim, 0.0)

    # 确保非负、对称
    sim = np.maximum(sim, 0.0)
    sim = 0.5 * (sim + sim.T)

    # 如果全部为 0，则退化为每个 domain 一个 cluster
    if np.all(sim == 0):
        print("⚠️ 当前 min_sim_th 下 domain 间无有效边，退化为每个 domain 单独一个 cluster。")
        domain_to_cluster = {d: i for i, d in enumerate(domains)}
    else:
        adjacency = csr_matrix(sim)

        # ---- 3. 构造“域级 AnnData”并跑 Leiden ----
        # 这里只用图结构，不关心 X 的具体值
        adata_dom = ad.AnnData(X=np.zeros((K, 1)))
        adata_dom.obs[domain_key] = domains
        adata_dom.obs_names = pd.Index(domains)

        # 把 adjacency 塞进 obsp，scanpy 会用它
        adata_dom.obsp["connectivities"] = adjacency

        sc.tl.leiden(
            adata_dom,
            resolution=resolution,
            key_added="leiden_domain",
            adjacency=adjacency,
        )

        # 每个 domain 的 Leiden cluster（字符串）
        dom_cluster = (adata_dom.obs["leiden_domain"].astype(int) + 1).astype(str)
        domain_to_cluster = dom_cluster.to_dict()  # {domain_name: cluster_id("1", "2", ...)}

    # ---- 4. 映射回每个 spot ----
    dom_series = adata.obs[domain_key].astype(str)
    cluster_labels = dom_series.map(domain_to_cluster)

    adata.obs[new_cluster_key] = cluster_labels.astype("int")
    adata.obs[new_cluster_key] = adata.obs[new_cluster_key].astype("category")

    # ---- 5. 保存 cluster -> domains 的映射（方便解释）----
    cluster_map = {}
    for d, cid in domain_to_cluster.items():
        cluster_map.setdefault(cid, []).append(d)
    # 排序一下
    cluster_map = {
        cid: sorted(dom_list)
        for cid, dom_list in sorted(cluster_map.items(), key=lambda x: x[0])
    }

    adata.uns[new_cluster_key + "_map"] = cluster_map
    print(f">>> Computed coarse domain clusters stored in adata.obs['{new_cluster_key}'].")
    print(f"Found {len(cluster_map)} coarse domain clusters.")
    print(f"    - Cluster to domains mapping stored in adata.uns['{new_cluster_key}_map'].")
    return cluster_map

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

def search_coarse_domains(
    adata,
    domain_key: str = "domain",
    min_sim_th: float = 0.1,
    resolution: float = 1.0,
    n_target: int = None,
):
    """
    在 domain graph 上对域点做聚类。
    新增 n_target 参数：若指定，则自动搜索最优 resolution 以匹配目标簇数。
    """
    new_cluster_key = f"{domain_key}_coarse"
    similarity_key = f"{domain_key}_similarity"

    if similarity_key not in adata.uns:
        raise KeyError(f"{similarity_key} 不在 adata.uns 中。")

    sim_df = adata.uns[similarity_key].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)
    domains = sim_df.index.to_list()
    K = len(domains)

    # ---- 1. 构建邻接矩阵 ----
    sim = sim_df.to_numpy()
    np.fill_diagonal(sim, 0.0)
    if min_sim_th > 0:
        sim = np.where(sim >= min_sim_th, sim, 0.0)
    sim = np.maximum(sim, 0.0)
    sim = 0.5 * (sim + sim.T)
    adjacency = csr_matrix(sim)

    # ---- 2. 构造域级 AnnData ----
    adata_dom = ad.AnnData(X=np.zeros((K, 1)))
    adata_dom.obs_names = pd.Index(domains)
    adata_dom.obsp["connectivities"] = adjacency

    # ---- 3. 核心聚类逻辑（支持 n_target 搜索） ----
    if n_target is not None:
        print(f"正在搜索最佳分辨率以匹配目标数量: {n_target}...")
        # 二分查找寻找合适的 resolution
        res_min, res_max = 0.01, 10.0
        best_res = resolution
        best_diff = float('inf')
        
        for i in range(20):  # 最多迭代20次
            current_res = (res_min + res_max) / 2
            sc.tl.leiden(adata_dom, resolution=current_res, key_added="temp", adjacency=adjacency)
            current_n = adata_dom.obs["temp"].nunique()
            
            if abs(current_n - n_target) < best_diff:
                best_diff = abs(current_n - n_target)
                best_res = current_res
            
            if current_n < n_target:
                res_min = current_res
            elif current_n > n_target:
                res_max = current_res
            else:
                best_res = current_res
                break
        final_res = best_res
    else:
        final_res = resolution

    # 执行最终聚类
    sc.tl.leiden(adata_dom, resolution=final_res, key_added="leiden_domain", adjacency=adjacency)
    
    # 转换为 1, 2, 3... 格式的字符串 ID
    dom_cluster = (adata_dom.obs["leiden_domain"].astype(int) + 1).astype(str)
    domain_to_cluster = dom_cluster.to_dict()

    # ---- 4. 映射回每个 spot ----
    dom_series = adata.obs[domain_key].astype(str)
    cluster_labels = dom_series.map(domain_to_cluster)

    adata.obs[new_cluster_key] = cluster_labels.astype("int")
    adata.obs[new_cluster_key] = adata.obs[new_cluster_key].astype("category")

    # ---- 5. 保存映射并打印结果 ----
    cluster_map = {}
    for d, cid in domain_to_cluster.items():
        cluster_map.setdefault(cid, []).append(d)
    
    cluster_map = {cid: sorted(dom_list) for cid, dom_list in sorted(cluster_map.items(), key=lambda x: int(x[0]))}
    adata.uns[new_cluster_key + "_map"] = cluster_map
    
    print(f">>> {new_cluster_key} 计算完成。最终分辨率: {final_res:.4f}")
    print(f"得到 {len(cluster_map)} 个粗领域簇。映射已存入 adata.uns['{new_cluster_key}_map']。")
    
    return cluster_map

import numpy as np

def compute_partition_entropy_from_labels(
    adata,
    labels,
    cons_key: str = "consensus_freq",
    normalized: bool = True,
):
    """
    给定一套 domain/cluster 标签，基于共识矩阵计算：
      - 每个 spot 的熵 H_i（可归一化）
      - 全局平均熵 H_global

    参数
    ----
    adata : AnnData
        需要包含 adata.obsm[cons_key]，形状 (N, N) 的共识矩阵。
    labels : array-like, shape (N,)
        当前每个 spot 的域标签（可以是 int 或 str）。
    cons_key : str
        共识矩阵在 obsm 中的 key，默认 "consensus_freq"。
    normalized : bool
        是否对熵做 log(K) 归一化，使得 H_i ∈ [0,1]。

    返回
    ----
    H_global : float
        全局平均熵。
    H_i : np.ndarray, shape (N,)
        每个 spot 的熵值。
    """

    C = adata.obsm[cons_key]
    if not isinstance(C, np.ndarray):
        C = np.asarray(C)
    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "consensus_freq 必须是 N×N 方阵"

    labels = np.asarray(labels).astype(str)

    # 有效域
    unique_domains = sorted(list(set(labels)))
    K = len(unique_domains)
    if K == 0:
        # 没有有效域，返回全 NaN
        H_i = np.full(N, np.nan, dtype=float)
        return np.nan, H_i

    # 为每个域收集索引
    domain_to_idx = {
        d: np.where(labels == d)[0] for d in unique_domains
    }

    # 计算 S(i,d) = mean_j∈域d C[i,j]
    S = np.zeros((N, K), dtype=float)
    for k, d in enumerate(unique_domains):
        idx_d = domain_to_idx[d]
        if len(idx_d) == 0:
            continue
        S[:, k] = C[:, idx_d].mean(axis=1)

    # 归一化成 P(i,d)
    row_sum = S.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = S / row_sum

    # 熵 H_i
    eps = 1e-12
    H_i = -np.sum(P * np.log(P + eps), axis=1)

    if normalized:
        H_i = H_i / (np.log(K + eps))

    H_global = float(np.nanmean(H_i))

    return H_global, H_i




import numpy as np
import pandas as pd
from itertools import combinations

def merge_clusters_by_DTI_fast(
    adata,
    domain_key: str = "fine",
    new_domain_key: str = "domain",
    target_n_domains: int = None,
    min_gain: float = 0.0,
    w_sim: float = 1.0,
    w_dti: float = 0.5,
    w_boundary: float = 0.5,
    compute_entropy: bool = False,
    entropy_every: int = 1,
    recompute_post: bool = True,
):
    """
    加速版：
    1. cluster similarity / boundary 用矩阵聚合增量更新
    2. cluster DTI 用加权和增量更新
    3. entropy 改成可选
    4. 后处理改成可选
    """

    adata.obs[domain_key] = adata.obs[domain_key].astype(str)

    # ---------- 0. base domains ----------
    sim_df = adata.uns[f"{domain_key}_similarity"].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)

    base_domains = sorted(set(adata.obs[domain_key].unique()) & set(sim_df.index))
    D = len(base_domains)
    if D == 0:
        raise ValueError(f"No valid domains found for {domain_key}")

    dom2idx = {d: i for i, d in enumerate(base_domains)}
    idx2dom = {i: d for i, d in enumerate(base_domains)}

    counts = adata.obs[domain_key].value_counts()
    base_size = np.array([float(counts.get(d, 0.0)) for d in base_domains], dtype=float)

    domain_dti_key = f"{domain_key}_DTI"
    if domain_dti_key in adata.obs.columns:
        df_dti = (
            adata.obs[[domain_key, domain_dti_key]]
            .dropna()
            .groupby(domain_key)[domain_dti_key]
            .mean()
        )
        base_dti = np.array([float(df_dti.get(d, 0.0)) for d in base_domains], dtype=float)
        base_dti_valid = np.array([d in df_dti.index for d in base_domains], dtype=bool)
    else:
        base_dti = np.zeros(D, dtype=float)
        base_dti_valid = np.zeros(D, dtype=bool)

    # ---------- 1. base-level sim / boundary matrices ----------
    sim_mat = sim_df.loc[base_domains, base_domains].to_numpy(dtype=float)

    # weighted sim numerator / denominator
    size_outer = np.outer(base_size, base_size)
    sim_num_base = sim_mat * size_outer
    sim_den_base = size_outer.copy()

    # boundary
    b_sum_base = np.zeros((D, D), dtype=float)
    b_cnt_base = np.zeros((D, D), dtype=float)

    boundary_df = adata.uns.get(f"{domain_key}_boundary_strength", pd.DataFrame())
    if len(boundary_df) > 0:
        for _, row in boundary_df.iterrows():
            a = str(row["domain_a"])
            b = str(row["domain_b"])
            s = float(row["strength"])
            if a in dom2idx and b in dom2idx:
                i, j = dom2idx[a], dom2idx[b]
                b_sum_base[i, j] = s
                b_sum_base[j, i] = s
                b_cnt_base[i, j] = 1.0
                b_cnt_base[j, i] = 1.0

    # ---------- 2. initialize clusters ----------
    # 每个 cluster 用一个 int id
    active = list(range(D))
    members = {i: [i] for i in range(D)}

    # cluster-level aggregated matrices
    sim_num = sim_num_base.copy()
    sim_den = sim_den_base.copy()
    b_sum = b_sum_base.copy()
    b_cnt = b_cnt_base.copy()

    # DTI incremental stats
    dti_weight_sum = {}
    cluster_size_sum = {}
    cluster_dti = {}
    for i in range(D):
        cluster_size_sum[i] = base_size[i]
        if base_dti_valid[i]:
            dti_weight_sum[i] = base_dti[i] * base_size[i]
            cluster_dti[i] = base_dti[i]
        else:
            dti_weight_sum[i] = np.nan
            cluster_dti[i] = np.nan

    # spot labels for final writeback
    base_labels = adata.obs[domain_key].astype(str).to_numpy()
    domain_to_cluster = {idx2dom[i]: i for i in range(D)}

    gain_log = []
    merge_id = D

    # entropy init
    global_entropy = np.nan
    if compute_entropy:
        cur_labels = np.array([domain_to_cluster.get(d, d) for d in base_labels])
        global_entropy, _ = compute_partition_entropy_from_labels(
            adata,
            labels=cur_labels,
            cons_key="consensus_freq",
            normalized=True,
        )

    # ---------- 3. greedy merge ----------
    step = 0
    while True:
        n_clusters = len(active)
        if target_n_domains is not None and n_clusters <= target_n_domains:
            break
        if n_clusters < 2:
            break

        best_gain = -np.inf
        best_pair = None

        # 枚举 cluster pair，但 pair 的 sim/boundary 读取已经是 O(1)
        for ii in range(n_clusters):
            c1 = active[ii]
            for jj in range(ii + 1, n_clusters):
                c2 = active[jj]

                den = sim_den[c1, c2]
                sim_val = sim_num[c1, c2] / den if den > 0 else 0.0

                dti1 = cluster_dti.get(c1, np.nan)
                dti2 = cluster_dti.get(c2, np.nan)
                if np.isnan(dti1) or np.isnan(dti2):
                    dti_term = 0.0
                else:
                    dti_term = 1.0 - max(dti1, dti2)

                cnt = b_cnt[c1, c2]
                b_val = b_sum[c1, c2] / cnt if cnt > 0 else 0.0

                gain = w_sim * sim_val + w_dti * dti_term - w_boundary * b_val

                if gain > best_gain:
                    best_gain = gain
                    best_pair = (c1, c2)

        if best_pair is None or best_gain < min_gain:
            break

        c1, c2 = best_pair
        new_c = merge_id
        merge_id += 1
        step += 1

        # 扩容矩阵
        old_n = sim_num.shape[0]
        new_n = old_n + 1
        sim_num = _expand_square(sim_num, new_n)
        sim_den = _expand_square(sim_den, new_n)
        b_sum = _expand_square(b_sum, new_n)
        b_cnt = _expand_square(b_cnt, new_n)

        # members
        members[new_c] = members[c1] + members[c2]

        # 增量更新：new_c 与其他 active cluster 的聚合值
        for x in active:
            if x in (c1, c2):
                continue
            sim_num[new_c, x] = sim_num[c1, x] + sim_num[c2, x]
            sim_num[x, new_c] = sim_num[new_c, x]

            sim_den[new_c, x] = sim_den[c1, x] + sim_den[c2, x]
            sim_den[x, new_c] = sim_den[new_c, x]

            b_sum[new_c, x] = b_sum[c1, x] + b_sum[c2, x]
            b_sum[x, new_c] = b_sum[new_c, x]

            b_cnt[new_c, x] = b_cnt[c1, x] + b_cnt[c2, x]
            b_cnt[x, new_c] = b_cnt[new_c, x]

        # self-block 置零
        sim_num[new_c, new_c] = 0.0
        sim_den[new_c, new_c] = 0.0
        b_sum[new_c, new_c] = 0.0
        b_cnt[new_c, new_c] = 0.0

        # DTI 增量更新
        size1 = cluster_size_sum[c1]
        size2 = cluster_size_sum[c2]
        cluster_size_sum[new_c] = size1 + size2

        d1 = dti_weight_sum.get(c1, np.nan)
        d2 = dti_weight_sum.get(c2, np.nan)
        if np.isnan(d1) and np.isnan(d2):
            dti_weight_sum[new_c] = np.nan
            cluster_dti[new_c] = np.nan
        else:
            v1 = 0.0 if np.isnan(d1) else d1
            v2 = 0.0 if np.isnan(d2) else d2
            dti_weight_sum[new_c] = v1 + v2
            cluster_dti[new_c] = dti_weight_sum[new_c] / cluster_size_sum[new_c]

        # 更新 base domain -> cluster
        for bi in members[new_c]:
            domain_to_cluster[idx2dom[bi]] = new_c

        # entropy 可选/降频
        entropy_before = np.nan
        entropy_after = np.nan
        entropy_delta = np.nan
        if compute_entropy and (step % entropy_every == 0):
            entropy_before = global_entropy
            cur_labels = np.array([domain_to_cluster.get(d, d) for d in base_labels])
            entropy_after, _ = compute_partition_entropy_from_labels(
                adata,
                labels=cur_labels,
                cons_key="consensus_freq",
                normalized=True,
            )
            entropy_delta = entropy_after - global_entropy
            global_entropy = entropy_after

        gain_log.append({
            "step": step,
            "n_clusters_before": n_clusters,
            "n_clusters_after": n_clusters - 1,
            "best_gain": float(best_gain),
            "best_pair": f"{c1}__{c2}",
            "entropy_before": float(entropy_before) if not np.isnan(entropy_before) else np.nan,
            "entropy_after": float(entropy_after) if not np.isnan(entropy_after) else np.nan,
            "entropy_delta": float(entropy_delta) if not np.isnan(entropy_delta) else np.nan,
        })

        # active set 更新
        active = [x for x in active if x not in (c1, c2)]
        active.append(new_c)

        # 清理旧 cluster 统计
        for old in (c1, c2):
            cluster_size_sum.pop(old, None)
            cluster_dti.pop(old, None)
            dti_weight_sum.pop(old, None)

    # ---------- 4. rename and write back ----------
    final_clusters = active[:]
    final_clusters_sorted = sorted(
        final_clusters,
        key=lambda c: cluster_size_sum[c],
        reverse=True
    )
    cluster_rename = {c: f"{i+1}" for i, c in enumerate(final_clusters_sorted)}

    meta_map = {
        cluster_rename[c]: sorted([idx2dom[i] for i in members[c]])
        for c in final_clusters_sorted
    }

    old_labels = adata.obs[domain_key].astype(str).tolist()
    new_labels_tmp = [domain_to_cluster.get(d, d) for d in old_labels]
    new_labels_final = [cluster_rename.get(x, x) for x in new_labels_tmp]

    adata.obs[new_domain_key] = pd.Categorical(new_labels_final)
    adata.obs[new_domain_key] = adata.obs[new_domain_key].astype(int).astype("category")

    adata.uns[new_domain_key + "_map"] = meta_map
    adata.uns[new_domain_key + "_gain_log"] = pd.DataFrame(gain_log)

    if recompute_post:
        compute_transition_scores_entropy(
            adata,
            domain_key=new_domain_key,
        )
        compute_domain_sim(
            adata,
            domain_key=new_domain_key,
        )

    return meta_map


def _expand_square(A, new_n):
    old_n = A.shape[0]
    if new_n <= old_n:
        return A
    B = np.zeros((new_n, new_n), dtype=A.dtype)
    B[:old_n, :old_n] = A
    return B