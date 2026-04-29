import igraph as ig
import leidenalg
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import ot
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from .plot import plot_coarse_scan
from tqdm import tqdm




def mclust_R(data, n_clusters, random_seed=42):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # import os
    # os.environ['R_HOME'] = 'E:\\R-4.4.1'
    modelNames = 'EEE'

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), n_clusters, modelNames)
    mclust_res = np.array(res[-2])
    
    return mclust_res

def consensus_clustering(adata, 
                         method='leiden', 
                         resolution_range=(1.0, 5.0, 0.1), 
                         n_neighbors=15, 
                         tau_ARI=0.9,
                         tau_k=3, 
                         use_stable='core', 
                         use_rep='X_pca', 
                         dims=25, 
                         radius=20, 
                         refinement=False, 
                         cal=False, 
                         plot=False):

    all_clusters = []         # Store all clustering results
    cluster_labels = []       # Store labels for DataFrame
    param_list = []           # Store each tested parameter value

    start_time = time.time()
    
    # Ensure neighbor graph is computed (Leiden only)
    if method == 'leiden':
        sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        metric='cosine'
        )
    print(f">>> Starting {method} clustering...")

    # Iterate through resolution or cluster number range
    for param in tqdm(np.arange(*resolution_range), desc=f"Running {method} clustering"):
        if method == 'leiden':
            sc.tl.leiden(adata, 
                         resolution=param, 
                         random_state=0,
                         flavor="leidenalg",
                         n_iterations=2,
                         directed=False)
            labels = adata.obs['leiden'].astype(int).values
        
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=int(param), random_state=0, n_init=10)
            labels = kmeans.fit_predict(adata.obsm[use_rep])
        
        elif method == 'mclust': 
            labels = mclust_R(data=adata.obsm[use_rep][:, :dims], n_clusters=int(param), random_seed=42)
        else:
            raise ValueError("Method must be 'leiden', 'kmeans', or 'mclust'.")

        all_clusters.append(labels)
        cluster_labels.append(labels)
        param_list.append(param)

    print(f">>> {method} clustering finished. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Store clustering results in a DataFrame
    clusters_df = pd.DataFrame(np.array(cluster_labels).T, index=adata.obs.index,
                               columns=[f"{method}_{p:.2f}" for p in param_list])
    
    # Select core interval
    print(">>> Auto select core interval...")
    R_sorted, metrics, similarities = compute_metrics_similarities(clusters_df)
    core, merged = find_core_stable_range(similarities, tau_ARI= tau_ARI, tau_k= tau_k)

    if use_stable == 'core':
        stable_set = core
    elif use_stable == 'merged':
        stable_set = merged
    elif use_stable == 'all':
        stable_set = [(resolution_range[0], resolution_range[1])]
    elif isinstance(use_stable, (tuple, list)) and len(use_stable) == 2:
        # 用户自定义一个区间
        stable_set = [tuple(use_stable)]
    else:
        raise ValueError("use_stable must be 'core', 'merged', 'all' or a (min,max) tuple/list")

    filtered_clusters, filtered_R_values = filter_clusters_by_intervals(all_clusters, resolution_range, stable_set)
    print(">>> Core interval select computed.")
    print(f"    Filtered {len(filtered_clusters)} clusterings "
        f"in core interval [{core[0]}, {core[1]}] ")
    if plot:
        plot_coarse_scan(R_sorted, metrics, similarities, intervals=stable_set)

    print(">>> Computing consensus matrix...")

    # Compute consensus matrix
    # Convert to 2D array: (n_clusters, n_cells)
    filtered_array = np.vstack(filtered_clusters)  # shape = (n_filtered_clusters, N)
    num_clusters, num_objects = filtered_array.shape
    pairwise_probabilities = np.zeros((num_objects, num_objects), dtype=np.float32)

    start_time = time.time()
    for i in range(num_objects):
        # Compare the i-th cell’s cluster assignments with all others
        cell_labels = filtered_array[:, i]  # shape (num_clusters,)
        # Count how many times each other cell shares the same cluster across all clustering runs
        equal_counts = np.sum(filtered_array == cell_labels[:, None], axis=0)
        pairwise_probabilities[i, :] = equal_counts / num_clusters
        pairwise_probabilities[i, i] = 0  # zero the diagonal

        if (i + 1) % max(1, num_objects // 10) == 0:
            print(f" Processed {i+1}/{num_objects} cells "
                f"({(i+1)/num_objects*100:.1f}%)")
            
    adata.obsm["consensus_freq"] = pairwise_probabilities
    print(f">>> Consensus matrix computed in {time.time() - start_time:.2f} seconds")

    # for labels in filtered_clusters:
    #     labels = np.array(labels)
    #     bool_matrix = labels[:, None] == labels
    #     similarity_matrix += bool_matrix

    # pairwise_probabilities = similarity_matrix / len(filtered_clusters)
    # np.fill_diagonal(pairwise_probabilities, 0)

    # Perform Leiden clustering on the consensus matrix
    if cal:
        print(">>> Performing final Leiden clustering on consensus matrix...")
        G = ig.Graph.Adjacency((pairwise_probabilities > 0).tolist())
        G.es['weight'] = pairwise_probabilities[pairwise_probabilities.nonzero()]
        partition = leidenalg.find_partition(G, leidenalg.SurpriseVertexPartition, weights='weight')
        print(">>> Consensus clustering completed.")
    
        con_domain = np.array(partition.membership)
        adata.obs["pre_domain"] = con_domain
        con_domain = adata.obs["pre_domain"].copy()
        if refinement:
            new_type = refine_label(adata, radius, key='pre_domain')
            adata.obs['pre_domain'] = new_type
        clusters_df[f'{method}_con'] = con_domain
        adata.obs["pre_domain"] = adata.obs["pre_domain"].astype("category")

    # Merge with existing clustering results if present
    if "clusters_results" in adata.obsm:
        adata.obsm["clusters_results"] = pd.concat([adata.obsm["clusters_results"], clusters_df], axis=1)
    else:
        adata.obsm["clusters_results"] = clusters_df
    if cal:
        print(">>> adata.obs['pre_domain'] generated!")
    print(">>> adata.obsm['consensus_freq'] generated!")
    print(">>> adata.obsm['clusters_results'] generated!")
    return

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]  
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def compute_metrics_similarities(clusters_df=None):
    """
    Step 2 — 计算每个分辨率的 metrics 与相邻分辨率的相似度

    Parameters
    ----------
    clusters_df : pd.DataFrame, optional
        每列是不同分辨率的聚类标签，index 对应样本
    all_clusters : list of arrays, optional
        每个元素是聚类标签数组，对应 R_sorted
    R_sorted : list or array, optional
        聚类参数（resolution 或簇数）排序列表

    Returns
    -------
    R_sorted : list
        排序后的聚类参数
    metrics : dict
        {r: {'labels': labels, 'k': num_clusters}, ...}
    similarities : dict
        {(r1,r2): {'ARI': value, 'Δk': value}, ...}
    """
    
    metrics = {}
    
    R_sorted = sorted(clusters_df.columns, key=lambda c: float(c.split('_')[1]))
    for r in R_sorted:
        labels = clusters_df[r].values
        k = len(np.unique(labels))
        metrics[r] = {'labels': labels, 'k': k}

    # 计算相邻分辨率的 ARI 与 Δk
    similarities = {}
    for i in range(len(R_sorted)-1):
        r1, r2 = R_sorted[i], R_sorted[i+1]
        labels1, labels2 = metrics[r1]['labels'], metrics[r2]['labels']
        ari = adjusted_rand_score(labels1, labels2)
        Δk = abs(metrics[r1]['k'] - metrics[r2]['k'])
        similarities[(r1, r2)] = {'ARI': ari, 'Δk': Δk}
    
    return R_sorted, metrics, similarities

def find_core_stable_range(similarities, tau_ARI=0.9, tau_k=1):
    """
    根据 similarities 字典，找核心稳定区间和所有稳定区间

    Parameters
    ----------
    similarities : dict
        {(r1_name, r2_name): {'ARI': val, 'Δk': val}, ...}
    tau_ARI : float
        ARI 阈值
    tau_k : int
        Δk 阈值
    expand : float
        可选，核心区间额外扩展宽度

    Returns
    -------
    core : tuple
        (core_start, core_end) 核心稳定区间
    merged : list of tuples
        所有稳定区间
    """

    # 1️⃣ 提取分辨率数值
    intervals = []
    for (r1_name, r2_name), vals in similarities.items():
        r1 = float(r1_name.split('_')[1])
        r2 = float(r2_name.split('_')[1])
        if vals['ARI'] >= tau_ARI and vals['Δk'] <= tau_k:
            intervals.append((r1, r2))

    if not intervals:
        return None, []

    # 2️⃣ 按起点排序
    intervals.sort(key=lambda x: x[0])

    # 3️⃣ 合并连续区间
    merged = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:  # 连续或相邻
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    # 4️⃣ 找最长区间作为核心
    core = max(merged, key=lambda x: x[1] - x[0])

    return core, merged

def filter_clusters_by_intervals(all_clusters, resolution_range, intervals):
    """
    根据已知 resolution_range 和核心/稳定区间筛选聚类结果

    Parameters
    ----------
    all_clusters : list of arrays
        每个元素是一个聚类标签数组，对应 resolution_range
    resolution_range : tuple
        (r_min, r_max, step)
    intervals : tuple or list of tuples
        单个核心区间 (start, end) 或多个稳定区间 [(start1, end1), ...]

    Returns
    -------
    filtered_clusters : list of arrays
        筛选出的聚类结果
    filtered_R_values : list of float
        对应的分辨率参数
    """
    # 统一 intervals 为列表形式
    if isinstance(intervals, tuple) and len(intervals) == 2 and all(isinstance(x, (int, float)) for x in intervals):
        interval_list = [intervals]
    elif isinstance(intervals, list):
        interval_list = intervals
    else:
        raise ValueError("intervals must be a tuple (core) or list of tuples (merged)")

    r_min, r_max, step = resolution_range
    R_values = np.round(np.arange(r_min, r_max + step, step), 5)  # 数值列表

    filtered_clusters = []
    filtered_R_values = []

    for r_val, labels in zip(R_values, all_clusters):
        for start, end in interval_list:
            if start <= r_val <= end:
                filtered_clusters.append(labels)
                filtered_R_values.append(r_val)
                break  # 属于任意区间即可
    return filtered_clusters, filtered_R_values