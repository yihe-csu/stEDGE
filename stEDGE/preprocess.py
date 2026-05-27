"""
preprocess.py ? stEDGE preprocessing and consensus clustering.

Consensus clustering uses parallel block-wise computation with optional memmap
support for large datasets. All multi-resolution clusterings are used without
stable-interval filtering.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
from typing import Optional, Tuple

import igraph as ig
import leidenalg
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from tqdm import tqdm


# ============================================================================
# 1. Lightweight single-resolution clustering (no full AnnData in worker)
# ============================================================================


def _build_expression_connectivities(
    X_rep: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
):
    """Build a spatial-expression neighbour graph for reuse across all resolutions.

    Parameters
    ----------
    X_rep : np.ndarray
        Representation matrix, shape (n_obs, n_features).
    n_neighbors : int
        Number of neighbours for the kNN graph.
    metric : str
        Distance metric for neighbour search.

    Returns
    -------
    scipy.sparse.csr_matrix
        Connectivities matrix from the neighbour graph.
    """
    import scipy.sparse as sp
    n_obs = X_rep.shape[0]
    ad_graph = sc.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32))
    ad_graph.obsm["X_rep"] = X_rep
    sc.pp.neighbors(ad_graph, use_rep="X_rep", n_neighbors=n_neighbors, metric=metric)
    return ad_graph.obsp["connectivities"].tocsr().copy()


def _run_single_clustering(
    param: float,
    method: str,
    X_rep: np.ndarray,
    connectivities=None,
    random_state: int = 0,
    leiden_flavor: str = "igraph",
    leiden_iterations: int = 2,
) -> Tuple[np.ndarray, float]:
    """Run a single clustering without passing a full AnnData into the worker.

    Parameters
    ----------
    param : float
        Resolution (Leiden) or number of clusters (KMeans).
    method : str
        ``"leiden"`` or ``"kmeans"``.
    X_rep : np.ndarray
        Representation matrix used by KMeans; shape for Leiden only.
    connectivities : scipy.sparse.csr_matrix or None
        Precomputed neighbour graph for Leiden (required when method='leiden').
    random_state : int
        Random seed.
    leiden_flavor : str
        Leiden flavour passed to scanpy.
    leiden_iterations : int
        Number of Leiden iterations.

    Returns
    -------
    labels : np.ndarray
        Integer cluster labels.
    param : float
        The parameter value (for sorting).
    """
    method = method.lower()

    if method == "leiden":
        if connectivities is None:
            raise ValueError("connectivities must be provided for Leiden.")
        n_obs = X_rep.shape[0]
        ad_tmp = sc.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32))
        ad_tmp.obsp["connectivities"] = connectivities
        sc.tl.leiden(
            ad_tmp,
            resolution=float(param),
            random_state=random_state,
            flavor=leiden_flavor,
            n_iterations=leiden_iterations,
            directed=False,
            obsp="connectivities",
            key_added="leiden",
        )
        labels = ad_tmp.obs["leiden"].astype(int).to_numpy()

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=int(param), random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_rep)

    else:
        raise ValueError("method must be 'leiden' or 'kmeans'.")

    return labels.astype(np.int32, copy=False), float(param)


# ============================================================================
# 2. Block-wise parallel consensus matrix
# ============================================================================


def _compute_consensus_block(
    filtered_array: np.ndarray,
    row_start: int,
    row_end: int,
) -> Tuple[int, np.ndarray]:
    """Compute a row-block of the consensus co-occurrence matrix in memory.

    Parameters
    ----------
    filtered_array : np.ndarray
        Stacked cluster labels, shape (K, N).
    row_start : int
        Start row index (inclusive).
    row_end : int
        End row index (exclusive).

    Returns
    -------
    row_start : int
        The start index for reassembly.
    block : np.ndarray
        Consensus block of shape (row_end - row_start, N).
    """
    K, N = filtered_array.shape
    block = np.empty((row_end - row_start, N), dtype=np.float32)
    for offset, i in enumerate(range(row_start, row_end)):
        labels_i = filtered_array[:, i]
        equal_counts = np.sum(filtered_array == labels_i[:, None], axis=0)
        block[offset, :] = equal_counts / K
        block[offset, i] = 0.0
    return row_start, block


def _compute_consensus_block_to_memmap(
    filtered_array: np.ndarray,
    row_start: int,
    row_end: int,
    memmap_path: str,
    shape: Tuple[int, int],
    dtype_name: str,
) -> Tuple[int, int]:
    """Compute a row-block of the consensus matrix and write to a memory-mapped file.

    Parameters
    ----------
    filtered_array : np.ndarray
        Stacked cluster labels, shape (K, N).
    row_start : int
        Start row index (inclusive).
    row_end : int
        End row index (exclusive).
    memmap_path : str
        Path to the memory-mapped output file.
    shape : tuple
        Shape of the full (N, N) consensus matrix.
    dtype_name : str
        NumPy dtype name for the memmap.

    Returns
    -------
    row_start : int
    row_end : int
    """
    K, N = filtered_array.shape
    out = np.memmap(memmap_path, dtype=dtype_name, mode="r+", shape=shape)
    block = np.empty((row_end - row_start, N), dtype=np.float32)
    for offset, i in enumerate(range(row_start, row_end)):
        labels_i = filtered_array[:, i]
        equal_counts = np.sum(filtered_array == labels_i[:, None], axis=0)
        block[offset, :] = equal_counts / K
        block[offset, i] = 0.0
    out[row_start:row_end, :] = block
    out.flush()
    return row_start, row_end


def compute_full_consensus_parallel(
    filtered_array: np.ndarray,
    n_jobs: int = -1,
    block_size: int = 512,
    use_memmap: bool = False,
    memmap_path: Optional[str] = None,
    max_dense_n: int = 100000,
    dtype=np.float32,
    backend: str = "loky",
) -> np.ndarray:
    """Compute the full N×N consensus co-occurrence matrix in parallel.

    Divides rows into blocks and dispatches to workers.  Supports in-memory
    and memory-mapped output for large datasets.

    Parameters
    ----------
    filtered_array : np.ndarray
        Stacked cluster labels, shape (K, N).
    n_jobs : int
        Number of parallel workers (-1 = all cores).
    block_size : int
        Number of rows per block.
    use_memmap : bool
        If True, write directly to a memory-mapped file.
    memmap_path : str or None
        Path for the memmap file (auto-generated if None).
    max_dense_n : int
        Raise MemoryError if N exceeds this and memmap is disabled.
    dtype : numpy dtype
        Output data type.
    backend : str
        joblib parallel backend.

    Returns
    -------
    np.ndarray or np.memmap
        Consensus matrix of shape (N, N).
    """
    filtered_array = np.asarray(filtered_array, dtype=np.int32)
    K, N = filtered_array.shape

    if N > max_dense_n and not use_memmap:
        estimated_gb = N * N * np.dtype(dtype).itemsize / 1024 ** 3
        raise MemoryError(
            f"N={N} is large for in-memory dense consensus. "
            f"Estimated matrix memory is {estimated_gb:.2f} GB. "
            "Set use_memmap=True or increase max_dense_n explicitly."
        )

    n_jobs_eff = multiprocessing.cpu_count() if n_jobs == -1 else max(1, int(n_jobs))
    row_blocks = [(s, min(s + block_size, N)) for s in range(0, N, block_size)]

    if use_memmap:
        if memmap_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="stedge_consensus_")
            memmap_path = os.path.join(tmp_dir, "consensus_freq.dat")
        consensus = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=(N, N))
        Parallel(n_jobs=n_jobs_eff, backend=backend)(
            delayed(_compute_consensus_block_to_memmap)(
                filtered_array, s, e, memmap_path, (N, N), np.dtype(dtype).name
            )
            for s, e in tqdm(row_blocks, desc="Computing consensus_freq")
        )
        consensus.flush()
        return consensus

    consensus = np.zeros((N, N), dtype=dtype)
    results = Parallel(n_jobs=n_jobs_eff, backend=backend)(
        delayed(_compute_consensus_block)(filtered_array, s, e)
        for s, e in tqdm(row_blocks, desc="Computing consensus_freq")
    )
    for row_start, block in results:
        consensus[row_start : row_start + block.shape[0], :] = block.astype(
            dtype, copy=False
        )
    return consensus


# ============================================================================
# 3. Optional final Leiden on consensus (cal=True)
# ============================================================================


def _run_final_leiden_on_consensus(
    consensus: np.ndarray,
    threshold: float = 0.0,
    partition_type: str = "surprise",
) -> np.ndarray:
    """Derive a hard partition from the consensus matrix via igraph Leiden.

    Parameters
    ----------
    consensus : np.ndarray
        N×N consensus co-occurrence matrix.
    threshold : float
        Edges with weight ≤ threshold are dropped.
    partition_type : str
        ``"surprise"`` or ``"modularity"``.

    Returns
    -------
    np.ndarray
        Integer cluster labels for each spot.
    """
    mask = consensus > threshold
    np.fill_diagonal(mask, False)
    sources, targets = np.nonzero(mask)
    weights = consensus[sources, targets].astype(float)
    graph = ig.Graph(
        n=consensus.shape[0],
        edges=list(zip(sources.tolist(), targets.tolist())),
        directed=False,
    )
    graph.es["weight"] = weights.tolist()

    if partition_type == "surprise":
        partition_cls = leidenalg.SurpriseVertexPartition
    elif partition_type == "modularity":
        partition_cls = leidenalg.ModularityVertexPartition
    else:
        raise ValueError("partition_type must be 'surprise' or 'modularity'.")
    partition = leidenalg.find_partition(graph, partition_cls, weights="weight")
    return np.asarray(partition.membership, dtype=np.int32)


# ============================================================================
# 4. Main: consensus_clustering
# ============================================================================


def consensus_clustering(
    adata,
    method: str = "leiden",
    resolution_range: Tuple[float, float, float] = (1.0, 2.0, 0.1),
    n_neighbors: int = 15,
    graph_key: Optional[str] = None,
    use_rep: str = "X_pca",
    dims: int = 25,
    radius: int = 20,
    refinement: bool = False,
    cal: bool = False,
    n_jobs: int = -1,
    metric: str = "cosine",
    random_state: int = 0,
    clustering_backend: str = "loky",
    consensus_backend: str = "loky",
    block_size: int = 512,
    use_memmap: bool = False,
    memmap_path: Optional[str] = None,
    max_dense_n: int = 100000,
    final_consensus_threshold: float = 0.0,
    final_partition_type: str = "surprise",
    leiden_flavor: str = "leidenalg",
):
    """
    Multi-resolution consensus clustering (parallel, block-wise).

    Runs the clustering method at every resolution in ``resolution_range``,
    builds a full co-occurrence consensus matrix, and optionally derives a
    hard partition via final Leiden on the consensus graph.

    Parameters
    ----------
    adata : AnnData
    method : str
        'leiden' or 'kmeans'.
    resolution_range : tuple
        (start, stop, step).
    n_neighbors : int
        Number of neighbors for the expression graph (Leiden only).
    graph_key : str or None
        Precomputed obsp key for the Leiden graph.  If None, builds one.
    use_rep : str
        Key in adata.obsm for the representation matrix.
    dims : int
        PCA dimensions (retained for compatibility; unused by leiden/kmeans).
    radius : int
        Neighbourhood radius for label refinement.
    refinement : bool
        Whether to apply spatial label refinement after final Leiden.
    cal : bool
        Whether to compute ``pre_domain`` via final Leiden.
    n_jobs : int
        Parallel workers (-1 = all cores).
    metric : str
        Distance metric for the expression neighbour graph.
    random_state : int
        Random seed.
    clustering_backend : str
        joblib backend for clustering.
    consensus_backend : str
        joblib backend for consensus computation.
    block_size : int
        Row block size for consensus computation.
    use_memmap : bool
        If True, write consensus to disk-backed memmap.
    memmap_path : str or None
        Path for memmap file.
    max_dense_n : int
        Safety threshold for in-memory dense consensus.
    final_consensus_threshold : float
        Edge threshold for final Leiden graph.
    final_partition_type : str
        'surprise' or 'modularity'.

    Returns
    -------
    adata : AnnData  (modified in-place)
    """
    start_time = time.time()
    method = method.lower()

    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm['{use_rep}'] not found.")

    X_rep = np.asarray(adata.obsm[use_rep])
    N = X_rep.shape[0]
    print(f">>> Starting {method} consensus clustering ({N} observations)...")

    # 1. Build graph once for Leiden
    connectivities = None
    if method == "leiden":
        if graph_key is not None:
            if graph_key not in adata.obsp:
                raise KeyError(
                    f"adata.obsp['{graph_key}'] not found. "
                    "Build the graph before calling consensus_clustering."
                )
            connectivities = adata.obsp[graph_key].tocsr().copy()
        else:
            connectivities = _build_expression_connectivities(
                X_rep=X_rep, n_neighbors=n_neighbors, metric=metric
            )

    # 2. Parallel multiresolution clustering
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else max(1, int(n_jobs))
    print(f">>> Using {n_cores} cores for parallel clustering...")

    param_range = np.arange(*resolution_range)
    if len(param_range) == 0:
        raise ValueError("resolution_range generated no parameter values.")

    results = Parallel(n_jobs=n_jobs, backend=clustering_backend)(
        delayed(_run_single_clustering)(
            param=param,
            method=method,
            X_rep=X_rep,
            connectivities=connectivities,
            random_state=random_state,
            leiden_flavor=leiden_flavor,
        )
        for param in tqdm(
            param_range,
            desc=f"Running {method} clustering",
            total=len(param_range),
        )
    )

    results = sorted(results, key=lambda x: x[1])
    all_clusters = [labels for labels, _ in results]
    param_list = [param for _, param in results]

    print(
        f">>> {method} clustering finished. "
        f"Time elapsed: {time.time() - start_time:.2f} seconds"
    )

    clusters_df = pd.DataFrame(
        np.asarray(all_clusters).T,
        index=adata.obs.index,
        columns=[f"{method}_{p:.2f}" for p in param_list],
    )

    # 3. Full consensus matrix (all clusterings)
    print(f">>> Computing full consensus matrix from {len(all_clusters)} clusterings...")
    consensus_start = time.time()

    filtered_array = np.vstack(all_clusters).astype(np.int32, copy=False)
    pairwise_probabilities = compute_full_consensus_parallel(
        filtered_array=filtered_array,
        n_jobs=n_jobs,
        block_size=block_size,
        use_memmap=use_memmap,
        memmap_path=memmap_path,
        max_dense_n=max_dense_n,
        dtype=np.float32,
        backend=consensus_backend,
    )

    adata.obsm["consensus_freq"] = pairwise_probabilities
    print(
        f">>> Consensus matrix computed in "
        f"{time.time() - consensus_start:.2f} seconds"
    )

    # 4. Optional final Leiden
    if cal:
        print(">>> Performing final Leiden clustering on consensus matrix...")
        con_domain = _run_final_leiden_on_consensus(
            pairwise_probabilities,
            threshold=final_consensus_threshold,
            partition_type=final_partition_type,
        )
        adata.obs["pre_domain"] = con_domain

        if refinement:
            new_type = refine_label(adata, radius, key="pre_domain", n_jobs=n_jobs)
            adata.obs["pre_domain"] = new_type

        clusters_df[f"{method}_con"] = con_domain
        adata.obs["pre_domain"] = adata.obs["pre_domain"].astype("category")
        print(">>> adata.obs['pre_domain'] generated!")

    # 5. Store results
    if "clusters_results" in adata.obsm:
        adata.obsm["clusters_results"] = pd.concat(
            [adata.obsm["clusters_results"], clusters_df], axis=1
        )
    else:
        adata.obsm["clusters_results"] = clusters_df

    # Metadata
    adata.uns["stedge_consensus"] = {
        "method": method,
        "resolution_range": list(resolution_range),
        "n_neighbors": n_neighbors,
        "graph_key": graph_key if graph_key is not None else "None",
        "use_rep": use_rep,
        "full_consensus": True,
        "n_clusterings": len(all_clusters),
        "use_memmap": use_memmap,
        "block_size": block_size,
        "metric": metric,
    }

    print(">>> adata.obsm['consensus_freq'] generated!")
    print(">>> adata.obsm['clusters_results'] generated!")
    print(
        f">>> Consensus clustering completed in "
        f"{time.time() - start_time:.2f} seconds"
    )
    return adata


# ============================================================================
# 5. Spatial label refinement
# ============================================================================


def refine_label(adata, radius=50, key="label", n_jobs=-1):
    """Spatial label refinement via neighbourhood majority voting.

    For each spot, replaces its label with the majority label among its
    ``radius`` nearest spatial neighbours.  Uses the POT library to compute
    the full pairwise Euclidean distance matrix (O(N²) memory).

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs[key]`` and ``obsm["spatial"]``.
    radius : int
        Number of spatial neighbours used for voting.
    key : str
        Key in ``adata.obs`` for the labels to refine.
    n_jobs : int
        Number of parallel workers (-1 = all cores).

    Returns
    -------
    list of str
        Refined labels.
    """
    n_neigh = radius
    old_type = adata.obs[key].values
    position = adata.obsm["spatial"]
    distance = ot.dist(position, position, metric="euclidean")
    n_cell = distance.shape[0]

    def _refine_single_cell(cell_idx):
        vec = distance[cell_idx, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            if j < len(index):
                neigh_type.append(old_type[index[j]])
        if neigh_type:
            return str(max(neigh_type, key=neigh_type.count))
        return str(old_type[cell_idx])

    new_type = Parallel(n_jobs=n_jobs)(
        delayed(_refine_single_cell)(i) for i in range(n_cell)
    )
    return new_type
