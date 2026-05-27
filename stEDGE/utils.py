import heapq
import os
from collections import defaultdict

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


def compute_boundary_probability(adata, n_neighbors=6):
    """
    Compute boundary probability (local inconsistency) for each spot.

    Stores the result in ``adata.obs["p_edge"]`` and ``adata.obs["a_p_edge"]``.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obsm["spatial"]`` and ``obsm["consensus_freq"]``.
    n_neighbors : int
        Number of spatial neighbours for the local agreement window.
    """
    coords = adata.obsm["spatial"]
    n_spots = coords.shape[0]
    consensus_freq = adata.obsm["consensus_freq"].copy()

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_indices = indices[:, 1:]  # exclude self

    inconsistency_scores = []
    for i in range(n_spots):
        neighbor_ids = neighbor_indices[i]
        local_consensus = consensus_freq[i, neighbor_ids]
        local_agreement = np.mean(local_consensus)
        local_inconsistency = 1 - local_agreement
        inconsistency_scores.append(local_inconsistency)

    # Raw inconsistency is already in [0,1] since consensus is in [0,1].
    # Clip instead of min-max scaling -- min-max destroys the genuine signal
    # by stretching low-variance distributions to span [0,1] in every dataset.
    p_edge = np.array(inconsistency_scores, dtype=np.float32)
    p_edge = np.clip(p_edge, 0.0, 1.0)

    adata.obs["p_edge"] = p_edge
    adata.obs["a_p_edge"] = 1.0 - p_edge

    print(">>> Computed boundary probabilities and stored in adata.obs['p_edge'].")


def compute_gradient(adata, norm=False, k=12, smooth=False):
    """
    Compute spatial gradient of p_edge at each spot using weighted least squares.

    Coordinate standardization is off by default to preserve physical distance
    relationships. Pre-smoothing is off by default to preserve narrow boundary
    signals. Closer neighbors receive higher weight in the least-squares fit.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["p_edge"]`` and ``obsm["spatial"]``.
    norm : bool
        Whether to standardize spatial coordinates (default False).
    k : int
        Number of spatial neighbors for gradient estimation (default 12).
    smooth : bool
        Whether to pre-smooth p_edge with local averaging (default False).
    """
    p_edge = adata.obs['p_edge'].values.copy()
    coords = adata.obsm['spatial'].copy()

    if norm:
        coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, indices = nbrs.kneighbors(coords)

    # Optional smoothing (off by default -- destroys narrow boundary signals)
    if smooth:
        sigma = np.median(dist) * 0.5
        p_edge_work = np.array([
            np.average(p_edge[nb], weights=np.exp(-0.5 * (d / sigma) ** 2))
            for d, nb in zip(dist, indices)
        ], dtype=np.float32)
    else:
        p_edge_work = p_edge

    grad = np.zeros_like(coords, dtype=np.float64)
    for i in range(len(coords)):
        neighbors = indices[i]
        delta_p = p_edge_work[neighbors] - p_edge_work[i]
        delta_coords = coords[neighbors] - coords[i]

        weights = np.exp(-0.5 * (dist[i] / (np.median(dist[i]) + 1e-8)) ** 2)
        W = np.diag(weights)

        try:
            A = delta_coords.T @ W @ delta_coords
            # Small ridge regularization for numerical stability
            A += 1e-6 * np.eye(A.shape[0])
            b = delta_coords.T @ W @ delta_p
            grad[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            grad[i] = np.linalg.lstsq(delta_coords, delta_p, rcond=None)[0]

    adata.obsm['grad_p_edge'] = grad
    print(">>> Computed gradient of p_edge and stored in adata.obsm['grad_p_edge'].")
    return


def find_seed_regions(
    adata,
    seed_quantile=0.10,
    n_neighbors=6,
    min_seed_size=5,
):
    """
    Seed region detection via global p_edge quantile.

    Selects the lowest ``seed_quantile`` fraction of p_edge spots as
    seed candidates, builds a spatial KNN graph among them, and
    extracts connected components.  Components smaller than
    ``min_seed_size`` are discarded.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["p_edge"]`` and ``obsm["spatial"]``.
    seed_quantile : float
        Fraction of lowest-p_edge spots used as candidates.
        Recommended: 0.05 – 0.15.
    n_neighbors : int
        Number of spatial neighbours for the seed connectivity graph.
    min_seed_size : int
        Minimum connected-component size to retain as a valid seed.

    Outputs
    -------
    adata.obs["labeled_seeds"]
    adata.obsm["knn_graph"]
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    p_edge = np.asarray(adata.obs["p_edge"].values, dtype=np.float64)
    n = len(p_edge)

    # ---- 1. Quantile threshold ----
    p_cap = float(np.quantile(p_edge, seed_quantile))
    seed_mask = p_edge <= p_cap
    low_p_indices = np.where(seed_mask)[0]

    # ---- 2. KNN graph (n_neighbors + 1 → remove self with [:, 1:]) ----
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    knn_graph = nn.kneighbors(coords, return_distance=False)[:, 1:]

    # ---- 3. Connected components among low-p_edge candidates ----
    low_p_set = set(low_p_indices.tolist())
    edges = [(int(i), int(j))
             for i in low_p_indices
             for j in knn_graph[i]
             if int(j) in low_p_set]
    G = nx.Graph()
    G.add_edges_from(edges)
    components = list(nx.connected_components(G))

    # ---- 4. Filter tiny fragments ----
    components = [c for c in components if len(c) >= min_seed_size]

    # ---- 5. Assign labels ----
    labeled_seeds = np.zeros(n, dtype=np.int32)
    for label, comp in enumerate(components, start=1):
        labeled_seeds[list(comp)] = label

    adata.obs["labeled_seeds"] = pd.Categorical(labeled_seeds)
    adata.obsm["knn_graph"] = knn_graph

    # ---- 6. Logging ----
    n_seed_points = int((labeled_seeds > 0).sum())
    if len(components) > 0:
        sizes = [len(c) for c in components]
        print(f">>> Found {len(components)} seed regions "
              f"(seed_quantile={seed_quantile:.2f}, p_cap={p_cap:.5f}).")
        print(f"    Seed points: {n_seed_points}/{n} ({n_seed_points / n:.3f})")
        print(f"    Seed size: min={np.min(sizes)}, "
              f"median={np.median(sizes):.1f}, max={np.max(sizes)}")
    else:
        print(">>> No valid seed regions found.")

    return


def region_growing_segmentation(
    adata,
    global_thr_k=0.25,
    jump_thr=None,
    jump_thr_quantile=0.95,
    use_global_cap=True,
    hole_fill=True,
    hole_fill_thr="global",
    output_dir=None,
    save_log=False,
):
    """
    Boundary-preserving competitive region growing.

    All seeds compete in a shared priority queue.  Expansion is gated by:
    - A per-seed threshold (capped by the global threshold when use_global_cap=True).
    - A local p_edge jump barrier (abs difference between parent and candidate).
    - An edge-barrier check (max of parent and candidate p_edge).

    Isolated holes (unlabeled spots surrounded by a single label) are optionally
    filled if their p_edge is below the relevant threshold.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["p_edge"]``, ``obs["labeled_seeds"]``,
        ``obsm["spatial"]``, and ``obsm["knn_graph"]``.
    global_thr_k : float
        IQR coefficient for thresholds (lower = tighter boundaries).
    jump_thr : float or None
        Max allowed local p_edge jump.  Estimated from KNN edge diffs if None.
    jump_thr_quantile : float
        Quantile used to estimate jump_thr.
    use_global_cap : bool
        If True, seed_threshold = min(seed_thr, global_thr).
    hole_fill : bool
        Whether to fill isolated unlabeled holes.
    hole_fill_thr : {"global", "seed"}
        Which threshold to use for hole filling.
    output_dir : str or None
        Directory for saving the growth log CSV (only used when save_log=True).
    save_log : bool
        Whether to record per-step expansion decisions in adata.uns.
    """
    coords = adata.obsm["spatial"]
    p_edge = np.asarray(adata.obs["p_edge"].values, dtype=np.float64)
    knn_graph = adata.obsm["knn_graph"]
    labeled_seeds = np.asarray(adata.obs["labeled_seeds"].values, dtype=int)
    grad_p_edge = adata.obsm.get("grad_p_edge", None)
    if grad_p_edge is not None:
        grad_p_edge = np.asarray(grad_p_edge, dtype=np.float64)
    n = len(p_edge)
    labels = np.zeros(n, dtype=np.int32)

    unique_seeds = np.unique(labeled_seeds)
    unique_seeds = unique_seeds[unique_seeds > 0]
    if len(unique_seeds) == 0:
        adata.obs["region_label"] = pd.Categorical(labels)
        adata.uns["region_growing_log"] = pd.DataFrame()
        print(">>> No seeds found. All labels set to 0.")
        return

    # ---- 1. Global & jump thresholds ----
    global_q1, global_q3 = np.percentile(p_edge, [25, 75])
    global_thr = global_q1 + global_thr_k * (global_q3 - global_q1)

    if jump_thr is None:
        diffs = []
        for i in range(n):
            for j in knn_graph[i]:
                if int(j) > i:
                    diffs.append(abs(p_edge[int(j)] - p_edge[i]))
        jump_thr = float(np.quantile(diffs, jump_thr_quantile)) if diffs else np.inf

    # ---- 2. Per-seed thresholds (conservative: capped by global) ----
    seed_thresholds = {}
    for sl in unique_seeds:
        seed_pe = p_edge[labeled_seeds == sl]
        if len(seed_pe) >= 3:
            sq1, sq3 = np.percentile(seed_pe, [25, 75])
            seed_thr = sq3 + global_thr_k * (sq3 - sq1)
        else:
            seed_thr = global_thr
        seed_thresholds[sl] = min(seed_thr, global_thr) if use_global_cap else seed_thr

    # ---- 3. Init: assign seeds, push frontiers ----
    heap = []
    in_queue = set()
    growth_log = [] if save_log else None

    for sl in unique_seeds:
        for i in np.where(labeled_seeds == sl)[0]:
            labels[i] = sl
            for j in knn_graph[i]:
                j = int(j)
                if labels[j] > 0:
                    continue
                key = (sl, j)
                if key in in_queue:
                    continue

                # Moving away from the boundary has lower priority and pops first.
                # grad_p_edge points toward increasing p_edge (toward boundary).
                # toward_boundary = max(0, dot(grad, direction)) > 0 when heading toward boundary.
                # Min-heap: 0 (away from boundary) pops before >0 (toward boundary).
                if grad_p_edge is not None:
                    direction = coords[j] - coords[i]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        direction = direction / norm
                        toward_boundary = max(0.0, np.dot(grad_p_edge[i], direction))
                    else:
                        toward_boundary = 0.0
                else:
                    toward_boundary = 0.0

                eb = max(p_edge[i], p_edge[j])

                heapq.heappush(
                    heap,
                    (
                        (toward_boundary, eb, p_edge[j], abs(p_edge[j] - p_edge[i]), sl),
                        sl,
                        i,
                        j,
                    ),
                )
                in_queue.add(key)

    # ---- 4. Competitive expansion ----
    while heap:
        (_, edge_bar, pe_j, local_jump, sl), seed_label, parent, j = heapq.heappop(heap)
        in_queue.discard((seed_label, j))

        if labels[j] > 0:
            continue

        thr = seed_thresholds[seed_label]
        accepted = pe_j <= thr and edge_bar <= thr and local_jump <= jump_thr

        if save_log:
            growth_log.append(dict(
                seed_label=seed_label, node=j, parent=parent,
                pe_node=pe_j, pe_parent=p_edge[parent],
                edge_barrier=edge_bar, local_jump=local_jump,
                seed_thr=thr, global_thr=global_thr,
                jump_thr=jump_thr, accepted=accepted,
            ))

        if not accepted:
            continue

        labels[j] = seed_label
        for k in knn_graph[j]:
            k = int(k)
            if labels[k] > 0:
                continue
            key = (seed_label, k)
            if key in in_queue:
                continue

            if grad_p_edge is not None:
                direction_k = coords[k] - coords[j]
                norm_k = np.linalg.norm(direction_k)
                if norm_k > 1e-8:
                    direction_k = direction_k / norm_k
                    toward_boundary_k = max(0.0, np.dot(grad_p_edge[j], direction_k))
                else:
                    toward_boundary_k = 0.0
            else:
                toward_boundary_k = 0.0

            eb_k = max(p_edge[j], p_edge[k])
            heapq.heappush(
                heap,
                (
                    (toward_boundary_k, eb_k, p_edge[k], abs(p_edge[k] - p_edge[j]), seed_label),
                    seed_label,
                    j,
                    k,
                ),
            )
            in_queue.add(key)

    # ---- 5. Conservative hole filling ----
    n_unlabeled_before = int((labels == 0).sum())
    filled = 0
    if hole_fill and n_unlabeled_before > 0:
        for i in np.where(labels == 0)[0]:
            nbr = labels[knn_graph[i]]
            nbr = nbr[nbr > 0]
            if len(nbr) == 0 or not np.all(nbr == nbr[0]):
                continue
            fl = int(nbr[0])
            thr = seed_thresholds.get(fl, global_thr) if hole_fill_thr == "seed" else global_thr
            if p_edge[i] <= thr:
                labels[i] = fl
                filled += 1

    # ---- 6. Write back ----
    adata.obs["region_label"] = pd.Categorical(labels)
    if save_log:
        log_df = pd.DataFrame(growth_log)
        adata.uns["region_growing_log"] = log_df
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_df.to_csv(os.path.join(output_dir, "region_growing_log.csv"), index=False)
    else:
        adata.uns["region_growing_log"] = pd.DataFrame()

    n_assigned = int((labels > 0).sum())
    print(f">>> Boundary-preserving region growing completed.")
    print(f"    Seeds: {len(unique_seeds)}  |  Assigned: {n_assigned}/{n}")
    print(f"    Global thr: {global_thr:.5f}  |  Jump thr: {jump_thr:.5f}")
    print(f"    Unlabeled → holes filled: {filled}  |  remaining: {n_unlabeled_before - filled}")
    print(f"    Labels stored in adata.obs['region_label'].")
    return


def propagate_domains_from_regions(
    adata,
    con_th=0.7,
    detect_unassigned=False,
    con_th_inner=0.7,
    min_size=20,
    con_region_key="con_region",
):
    """
    Consensus-based domain propagation from region_label seeds.

    For each region (processed largest-first), computes the mean consensus
    of every spot with region members.  Spots whose mean consensus exceeds
    con_th are assigned, along with unassigned spots fully spatially
    surrounded by the cluster.  Isolated assigned spots (no neighbour in
    the cluster) are removed.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["region_label"]``, ``obsm["consensus_freq"]``,
        and ``obsm["spatial"]``.
    con_th : float
        Mean-consensus threshold for cluster membership.
    detect_unassigned : bool
        Whether to detect new domains from remaining unassigned spots.
    con_th_inner : float
        Consensus threshold for new-domain detection.
    min_size : int
        Minimum size for new domains.
    con_region_key : str
        Key in ``adata.obs`` for storing propagated domain labels.
    """
    region_labels = adata.obs["region_label"].values
    consensus_freq = adata.obsm["consensus_freq"]
    n_spots = consensus_freq.shape[0]

    con_region = np.zeros(n_spots, dtype=int)
    assigned = set()
    current_label = 1

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

        cluster_scores = consensus_freq[region_indices].mean(axis=0)
        if hasattr(cluster_scores, "A1"):
            cluster_scores = np.asarray(cluster_scores).ravel()

        cluster_indices = [
            idx for idx in np.where(cluster_scores >= con_th)[0]
            if idx not in assigned
        ]
        if not cluster_indices:
            continue

        # Absorb unassigned spots fully surrounded by this cluster
        cluster_set = set(cluster_indices)
        unassigned_arr = np.array([
            i for i in range(n_spots)
            if i not in assigned and i not in cluster_set
        ])
        if len(unassigned_arr) > 0:
            _, neighbors = nbrs.kneighbors(coords[unassigned_arr])
            neighbors = neighbors[:, 1:]
            for i, neigh in enumerate(neighbors):
                if all(n in cluster_set for n in neigh):
                    cluster_set.add(int(unassigned_arr[i]))

        # Remove isolated spots (no neighbour in cluster)
        cluster_arr = np.array(list(cluster_set))
        _, neighbors_iso = nbrs.kneighbors(coords[cluster_arr])
        neighbors_iso = neighbors_iso[:, 1:]
        isolated = []
        for i, neigh in enumerate(neighbors_iso):
            if all(n not in cluster_set for n in neigh):
                isolated.append(int(cluster_arr[i]))
        cluster_set.difference_update(isolated)

        if not cluster_set:
            continue

        for idx in cluster_set:
            con_region[idx] = current_label
        assigned.update(cluster_set)
        current_label += 1

    adata.obs[con_region_key] = pd.Categorical(con_region)

    n_assigned = int((con_region > 0).sum())
    print(">>> Completed domain propagation by Con.")
    print(f"    - {current_label - 1} domains, {n_assigned}/{n_spots} spots assigned.")
    print(f"    - Con domain labels stored in adata.obs['{con_region_key}'].")

    if detect_unassigned:
        print(">>> Detecting new domains from unassigned points.")
        detect_new_domains_from_unassigned(
            adata,
            con_th_inner=con_th_inner,
            min_size=min_size,
            con_region_key=con_region_key,
        )

    return


def detect_new_domains_from_unassigned(
    adata,
    con_th_inner=0.7,
    min_size=20,
    con_region_key="con_region",
):
    """
    Detect new domains from unassigned spots via sparse connectivity graph.

    Builds a thresholded adjacency among unassigned spots, extracts connected
    components, and assigns new domain labels to components meeting min_size.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obsm["consensus_freq"]``.
    con_th_inner : float
        Consensus threshold for connecting unassigned spots.
    min_size : int
        Minimum connected-component size to retain as a new domain.
    con_region_key : str
        Key in ``adata.obs`` for the domain labels (updated in place).
    """
    con_region = adata.obs[con_region_key].astype(int).values.copy()
    unassigned = np.flatnonzero(con_region == 0)
    if len(unassigned) == 0:
        print("  - No unassigned points to process.")
        return

    consensus_freq = adata.obsm["consensus_freq"]
    sub_freq = consensus_freq[unassigned][:, unassigned]
    if not sparse.issparse(sub_freq):
        adj = sparse.csr_matrix(sub_freq >= con_th_inner)
    else:
        adj = (sub_freq >= con_th_inner).astype(np.uint8).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    n_components, comp_labels = connected_components(adj, directed=False, connection="weak")

    max_label = int(con_region.max())
    new_count = 0
    for comp_id in range(n_components):
        local_idx = np.flatnonzero(comp_labels == comp_id)
        if len(local_idx) < min_size:
            continue
        max_label += 1
        new_count += 1
        con_region[unassigned[local_idx]] = max_label

    adata.obs[con_region_key] = pd.Categorical(con_region)
    print(f"  - New domains detected from unassigned points: {new_count}")
    return


def label_spreading_with_pca(
    adata,
    n_components=50,
    n_neighbors=7,
    alpha=0.8,
    spatial_weight=0.3,
):
    """
    PCA-based LabelSpreading with spatial constraint.

    Spatial coordinates are concatenated with PCA features so the kNN graph
    respects both expression similarity and spatial proximity. This prevents
    labels from jumping across non-adjacent tissue regions.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["con_region"]``, ``obsm["spatial"]``, and ``.X``.
    n_components : int
        Number of PCA components (default 50).
    n_neighbors : int
        kNN neighbors for the LabelSpreading kernel (default 7).
    alpha : float
        LabelSpreading clamping factor (default 0.8).
    spatial_weight : float
        Weight of spatial coordinates relative to expression PCs.
        0 = expression-only, 0.3 = default balance.
    """
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(adata.X)

    if spatial_weight > 0:
        coords = adata.obsm["spatial"].copy()
        coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        X_combined = np.hstack([X_pca, spatial_weight * coords_norm * np.std(X_pca)])
    else:
        X_combined = X_pca

    labels = adata.obs["con_region"].astype(int).values
    y = np.where(labels == 0, -1, labels)

    model = LabelSpreading(
        kernel='knn',
        n_neighbors=n_neighbors,
        alpha=alpha,
        max_iter=1000
    )
    model.fit(X_combined, y)

    adata.obs["fine"] = model.transduction_

    return model


def label_propagation_with_pca(
    adata,
    n_components=50,
    n_neighbors=7,
    spatial_weight=0.3,
):
    """
    PCA-based LabelPropagation with spatial constraint.

    Spatial coordinates are concatenated with PCA features so the kNN graph
    respects both expression similarity and spatial proximity. This prevents
    labels from jumping across non-adjacent tissue regions.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs["con_region"]``, ``obsm["spatial"]``, and ``.X``.
    n_components : int
        Number of PCA components (default 50).
    n_neighbors : int
        kNN neighbors for LabelPropagation (default 7).
    spatial_weight : float
        Weight of spatial coordinates relative to expression PCs.
        0 = expression-only, 0.3 = balanced default.
    """
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(adata.X)

    if spatial_weight > 0:
        coords = adata.obsm["spatial"].copy()
        coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        X_combined = np.hstack([X_pca, spatial_weight * coords_norm * np.std(X_pca)])
    else:
        X_combined = X_pca

    orig_labels = adata.obs["con_region"].astype(int).values
    y = np.where(orig_labels == 0, -1, orig_labels)

    model = LabelPropagation(
        kernel='knn',
        n_neighbors=n_neighbors,
        max_iter=1000
    )
    model.fit(X_combined, y)

    new_labels = model.transduction_.copy()
    mask = orig_labels != 0
    new_labels[mask] = orig_labels[mask]

    adata.obs["fine"] = new_labels.astype(int)
    adata.obs["fine"] = adata.obs["fine"].astype("category")
    adata.obs["fine"] = adata.obs["fine"].cat.rename_categories(str)
    print(">>> Completed domain propagation with LabelPropagation (spatially constrained).")
    print("    - Propagated labels stored in adata.obs['fine'].")

    return model


def compute_transition_scores_entropy(
    adata,
    cons_key: str = "consensus_freq",
    domain_key: str = "domain",
    eps: float = 1e-12,
):
    """
    Compute entropy-based transition scores and Domain Transition Index.

    For each spot, computes soft assignment probabilities P(i,d) from the
    consensus matrix, then derives:

    - **TS_entropy(i)** = H(P(i,·)) / log(K), where H is Shannon entropy.
      Normalized to [0,1]: near 0 = confident assignment (one-hot-like),
      near 1 = high mixing across domains (transitional / uncertain).
    - **DTI_entropy(d)** = mean TS_entropy(i) for spots in domain d.

    Stores results in ``adata.obs``, soft probabilities in ``adata.obsm``,
    and domain-level DTI in ``adata.uns``.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obsm[cons_key]`` (N×N consensus) and
        ``obs[domain_key]`` (domain labels).
    cons_key : str
        Key for the consensus matrix in ``obsm``.
    domain_key : str
        Key for domain labels in ``obs``.
    eps : float
        Small constant to avoid log(0) in entropy computation.
    """

    C = adata.obsm[cons_key]
    if not isinstance(C, np.ndarray):
        C = np.asarray(C)

    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "consensus_freq must be N×N"

    domains_str = adata.obs[domain_key].astype(str).to_numpy()
    unique_domains = sorted(set(domains_str))
    domain_to_idx = {d: np.where(domains_str == d)[0] for d in unique_domains}

    K = len(unique_domains)
    S = np.zeros((N, K), dtype=float)
    for k, d in enumerate(unique_domains):
        idx_d = domain_to_idx[d]
        if len(idx_d) == 0:
            continue
        S[:, k] = C[:, idx_d].mean(axis=1)

    row_sum = S.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = S / row_sum

    P_clamped = np.clip(P, eps, 1.0)
    logP = np.log(P_clamped)
    H = -np.sum(P_clamped * logP, axis=1)

    if K > 1:
        TS_entropy = H / np.log(K)
        TS_entropy = np.clip(TS_entropy, 0.0, 1.0)
    else:
        TS_entropy = np.zeros_like(H)

    adata.obs[f"{domain_key}_TS"] = TS_entropy

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

    domain_DTI_list = [domain_DTI_map.get(d, np.nan) for d in domains_str]
    adata.obs[f"{domain_key}_DTI"] = np.array(domain_DTI_list, dtype=float)

    adata.obsm[f"{domain_key}_soft_affinity"] = S
    adata.obsm[f"{domain_key}_soft_prob"] = P
    adata.uns[f"{domain_key}_soft_domains"] = np.array(unique_domains, dtype=object)
    adata.uns[f"{domain_key}_DTI"] = df_dti

    return


def compute_domain_sim(
    adata,
    domain_key: str = "domain",

):
    """
    Build domain similarity and distance matrices from adata.obsm['domain_soft_prob'].

    Similarity is defined as:
        sim(a, b) = mean_i [P(i, a) * P(i, b)]  (shared spot assignment)

    Then normalize to [0, 1] and define distance as:
        dist(a,b) = 1 - sim_norm(a,b)

    Results are written to:
      - adata.uns[similarity_key] : pd.DataFrame indexed by domain name
      - adata.uns[distance_key]   : same index and columns
    """

    prob_key=f"{domain_key}_soft_prob"
    soft_domains_key=f"{domain_key}_soft_domains"
    distance_key=f"{domain_key}_distance"
    similarity_key=f"{domain_key}_similarity"

    P = adata.obsm[prob_key]                    # Shape (N, K)
    domains = np.array(adata.uns[soft_domains_key], dtype=str)  # Length K

    # Co-occurrence matrix: sum_i P(i, a) * P(i, b)
    S = P.T @ P    # Shape (K, K)
    N = P.shape[0]
    S = S / N      # Convert to mean_i[P(i, a) * P(i, b)]

    # Optionally drop the diagonal.
    np.fill_diagonal(S, 0.0)

    # Normalize to [0, 1].
    upper = S[np.triu_indices_from(S, k=1)]
    max_val = upper.max() if upper.size > 0 else 1.0
    if max_val == 0:
        S_norm = S
    else:
        S_norm = S / max_val

    # Similarity matrix: larger values mean greater similarity.
    sim_df = pd.DataFrame(S_norm, index=domains, columns=domains)

    # Distance matrix: larger values mean greater distance.
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
    Compute domain adjacency and boundary strength from spatial KNN graph.

    For each pair of adjacent domains, records the mean p_edge along their
    spatial interface as the boundary strength. Stored in ``adata.uns`` as
    a DataFrame with columns ``domain_a``, ``domain_b``, ``strength``.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obsm["spatial"]`` and ``obs[domain_key]``.
        If ``obs["p_edge"]`` exists, it is used for boundary strength.
    domain_key : str
        Key for domain labels in ``obs``.
    n_neighbors : int
        Number of spatial neighbors for the KNN graph.
    uns_key : str or None
        Key for storing the boundary strength DataFrame in ``adata.uns``.
        Defaults to ``f"{domain_key}_boundary_strength"``.
    """
    coords = adata.obsm["spatial"]
    dom = adata.obs[domain_key].to_numpy()
    n = coords.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_indices = indices[:, 1:]

    adj_pairs = set()
    boundary_pairs = defaultdict(list)
    p_edge = adata.obs["p_edge"].to_numpy() if "p_edge" in adata.obs.columns else None

    for i in range(n):
        di = dom[i]
        for j in neighbor_indices[i]:
            dj = dom[j]
            if di == dj:
                continue
            a, b = sorted([str(di), str(dj)])
            pair = (a, b)
            adj_pairs.add(pair)
            if p_edge is not None:
                boundary_pairs[pair].append((p_edge[i] + p_edge[j]) / 2)

    boundary_strength_data = []
    for (d1, d2), vals in boundary_pairs.items():
        boundary_strength_data.append({
            "domain_a": d1,
            "domain_b": d2,
            "strength": float(np.mean(vals)) if vals else 0.5,
        })
    # Placeholder row ensures downstream consumers always see required columns
    if not boundary_strength_data:
        boundary_strength_data = [
        {"domain_a": "__placeholder__", "domain_b": "__placeholder__", "strength": 0.0}
    ]

    if uns_key is None:
        uns_key = f"{domain_key}_boundary_strength"

    df_strength = pd.DataFrame(boundary_strength_data)
    adata.uns[uns_key] = df_strength

    print(f">>> Adjacency pairs calculated. Boundary strength stored in adata.uns['{uns_key}'].")

    return


def search_coarse_domains(
    adata,
    domain_key: str = "domain",
    min_sim_threshold: float = None,
    min_sim_quantile: float = 0.65,
    resolution: float = 1.0,
    n_target: int = None,
    random_state: int = 0,
):
    """Leiden clustering on the domain similarity graph."""
    new_cluster_key = f"{domain_key}_coarse"
    similarity_key = f"{domain_key}_similarity"
    if similarity_key not in adata.uns:
        raise KeyError(f"{similarity_key} not in adata.uns.")

    sim_df = adata.uns[similarity_key].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)
    domains = sim_df.index.to_list()
    K = len(domains)

    # ---- 1. Build adjacency with priority thresholding ----
    sim = sim_df.to_numpy(dtype=float)
    np.fill_diagonal(sim, 0.0)
    sim = np.maximum(sim, 0.0)

    # Prefer absolute thresholds.
    if min_sim_threshold is not None:
        # 1. Use explicit thresholds before quantiles.
        thr = float(min_sim_threshold)
    else:
        # 2. Otherwise use the default quantile for adaptive filtering.
        thr = 0.0
        if min_sim_quantile is not None and K > 1:
            upper = sim[np.triu_indices_from(sim, k=1)]
            if upper.size > 0:
                thr = float(np.quantile(upper, min_sim_quantile))

    sim[sim < thr] = 0.0

    # Force symmetry so Leiden uses an undirected graph.
    sim = 0.5 * (sim + sim.T)

    adjacency = csr_matrix(sim)

    # ---- 2. Build domain-level AnnData ----
    adata_dom = ad.AnnData(X=np.zeros((K, 1)))
    adata_dom.obs_names = pd.Index(domains)
    adata_dom.obsp["connectivities"] = adjacency

    # ---- 3. Leiden (with optional n_target search) ----
    if n_target is not None and n_target < K:
        best_res = resolution
        best_diff = K
        for r in np.linspace(0.1, 5.0, 30):
            sc.tl.leiden(adata_dom, resolution=float(r),
                         key_added="_tmp", adjacency=adjacency,
                         random_state=random_state)
            nc = adata_dom.obs["_tmp"].nunique()
            if abs(nc - n_target) < best_diff:
                best_diff = abs(nc - n_target)
                best_res = float(r)
            if nc == n_target:
                break
        final_res = best_res
    else:
        final_res = resolution

    sc.tl.leiden(adata_dom, resolution=final_res,
                 key_added="leiden_domain", adjacency=adjacency,
                 random_state=random_state)

    # ---- 4. Map back to spots ----
    dom_cluster = (adata_dom.obs["leiden_domain"].astype(int) + 1).astype(str)
    domain_to_cluster = dom_cluster.to_dict()

    cluster_labels = adata.obs[domain_key].astype(str).map(domain_to_cluster)
    adata.obs[new_cluster_key] = cluster_labels.astype("int").astype("category")
    adata.obs[new_cluster_key] = adata.obs[new_cluster_key].cat.rename_categories(str)

    # ---- 5. Save mapping ----
    cluster_map = {}
    for d, cid in domain_to_cluster.items():
        cluster_map.setdefault(cid, []).append(d)
    cluster_map = {cid: sorted(lst) for cid, lst in
                   sorted(cluster_map.items(), key=lambda x: int(x[0]))}
    adata.uns[new_cluster_key + "_map"] = cluster_map

    # Log the threshold mode used.
    mode = "absolute" if min_sim_threshold is not None else f"quantile({min_sim_quantile})"
    print(
        f">>> {new_cluster_key}: {len(cluster_map)} coarse domains "
        f"(resolution={final_res:.4f}, thr={thr:.4f} by {mode})."
    )
    return cluster_map


def compute_partition_entropy_from_labels(
    adata,
    labels,
    cons_key: str = "consensus_freq",
    normalized: bool = True,
):
    """
    Compute per-spot partition entropy from a consensus matrix and domain labels.

    For each spot, computes soft assignment probabilities P(i,d) by averaging
    consensus with members of each domain.  The entropy H(i) = -∑ P log P
    measures how mixed a spot's consensus pattern is across domains.
    Normalized by log(K) so the result falls in [0,1].

    Parameters
    ----------
    adata : AnnData
        Must contain ``obsm[cons_key]``, an N×N consensus matrix.
    labels : array-like, shape (N,)
        Domain labels for each spot (int or str).
    cons_key : str
        Key for the consensus matrix in ``obsm``.
    normalized : bool
        If True, divide by log(K) so H_i ∈ [0,1].

    Returns
    -------
    H_global : float
        Mean entropy across all spots.
    H_i : np.ndarray, shape (N,)
        Per-spot entropy values.
    """
    C = adata.obsm[cons_key]
    if not isinstance(C, np.ndarray):
        C = np.asarray(C)
    N = C.shape[0]
    assert C.shape[0] == C.shape[1], "consensus_freq must be N×N"

    labels = np.asarray(labels).astype(str)
    unique_domains = sorted(set(labels))
    K = len(unique_domains)
    if K == 0:
        H_i = np.full(N, np.nan, dtype=float)
        return np.nan, H_i

    domain_to_idx = {d: np.where(labels == d)[0] for d in unique_domains}

    S = np.zeros((N, K), dtype=float)
    for k, d in enumerate(unique_domains):
        idx_d = domain_to_idx[d]
        if len(idx_d) == 0:
            continue
        S[:, k] = C[:, idx_d].mean(axis=1)

    row_sum = S.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = S / row_sum

    eps = 1e-12
    H_i = -np.sum(P * np.log(P + eps), axis=1)
    if normalized:
        H_i = H_i / np.log(K + eps)

    H_global = float(np.nanmean(H_i))
    return H_global, H_i


def merge_clusters_by_DTI(
    adata,
    domain_key: str = "fine",
    new_domain_key: str = "domain",
    target_n_domains: int = None,
    min_gain: float = 1.0,
    w_sim: float = 1.0,
    w_dti: float = 0.5,
    w_boundary: float = 0.5,
    compute_entropy: bool = False,
    entropy_every: int = 1,
    recompute_post: bool = True,
    max_dti_threshold: float = 0.65,
):
    """
    Greedy hierarchical domain merging guided by DTI, similarity, and boundary.

    Modes of operation:
    1. Force Target Mode: If `target_n_domains` is provided, quality gates
       (`min_gain` and `max_dti_threshold`) are ignored to force merging down to the target.
    2. Data-Driven Mode: If `target_n_domains` is None, merging stops dynamically
       when no pair exceeds `min_gain` or if the merge exceeds `max_dti_threshold`.
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

    size_outer = np.outer(base_size, base_size)
    sim_num_base = sim_mat * size_outer
    sim_den_base = size_outer.copy()

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
    active = list(range(D))
    members = {i: [i] for i in range(D)}

    sim_num = sim_num_base.copy()
    sim_den = sim_den_base.copy()
    b_sum = b_sum_base.copy()
    b_cnt = b_cnt_base.copy()

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

    base_labels = adata.obs[domain_key].astype(str).to_numpy()
    domain_to_cluster = {idx2dom[i]: i for i in range(D)}

    gain_log = []
    merge_id = D

    global_entropy = np.nan
    if compute_entropy:
        cur_labels = np.array([domain_to_cluster.get(d, d) for d in base_labels])
        # Assume compute_partition_entropy_from_labels is in scope.
        global_entropy, _ = compute_partition_entropy_from_labels(
            adata,
            labels=cur_labels,
            cons_key="consensus_freq",
            normalized=True,
        )

    # ---------- 3. Fixed (mean, std) normalizers ----------
    def _raw_terms(c1, c2):
        den = sim_den[c1, c2]
        sv = sim_num[c1, c2] / den if den > 0 else 0.0
        d1 = cluster_dti.get(c1, np.nan)
        d2 = cluster_dti.get(c2, np.nan)
        dt = 0.0 if (np.isnan(d1) or np.isnan(d2)) else (1.0 - max(d1, d2))
        cnt = b_cnt[c1, c2]
        bv = b_sum[c1, c2] / cnt if cnt > 0 else 0.0
        return sv, dt, bv

    base_sv, base_dt, base_bv = [], [], []
    for ii in range(D):
        for jj in range(ii + 1, D):
            sv, dt, bv = _raw_terms(ii, jj)
            base_sv.append(sv)
            base_dt.append(dt)
            base_bv.append(bv)

    def _mean_std(arr):
        a = np.asarray(arr, dtype=float)
        m, s = float(np.mean(a)), float(np.std(a))
        return m, (s if s > 1e-12 else 1.0)

    sim_mu, sim_sig = _mean_std(base_sv)
    dti_mu, dti_sig = _mean_std(base_dt)
    b_mu,   b_sig   = _mean_std(base_bv)

    # ---------- 4. Greedy merge loop ----------
    step = 0
    # Check whether forced target-count mode is enabled.
    is_force_target_mode = (target_n_domains is not None)

    while True:
        n_clusters = len(active)

        # Stop once the target count is reached.
        if target_n_domains is not None and n_clusters <= target_n_domains:
            break
        if n_clusters < 2:
            break

        best_gain = -np.inf
        best_pair = None
        for ii in range(n_clusters):
            c1 = active[ii]
            for jj in range(ii + 1, n_clusters):
                c2 = active[jj]
                sv, dt, bv = _raw_terms(c1, c2)
                z_sim = (sv - sim_mu) / sim_sig
                z_dti = (dt - dti_mu) / dti_sig
                z_b   = (bv - b_mu)   / b_sig
                gain = w_sim * z_sim + w_dti * z_dti - w_boundary * z_b

                if gain > best_gain:
                    best_gain = gain
                    best_pair = (c1, c2)

        # Stop if no merge candidates remain.
        if best_pair is None:
            break

        # Gate 1: stop on low gain only outside forced target mode.
        if not is_force_target_mode and best_gain < min_gain:
            break

        c1, c2 = best_pair

        # ---- Post-merge quality gate ----
        s1, s2 = cluster_size_sum[c1], cluster_size_sum[c2]
        w1 = dti_weight_sum.get(c1, np.nan)
        w2 = dti_weight_sum.get(c2, np.nan)
        if not (np.isnan(w1) and np.isnan(w2)):
            v1 = 0.0 if np.isnan(w1) else w1
            v2 = 0.0 if np.isnan(w2) else w2
            merged_dti_preview = (v1 + v2) / (s1 + s2) if (s1 + s2) > 0 else 0.0

            # Gate 2: stop on high heterogeneity only outside forced target mode.
            if not is_force_target_mode and merged_dti_preview > max_dti_threshold:
                break

        # ---- Execute merge logic ----
        new_c = merge_id
        merge_id += 1
        step += 1

        old_n = sim_num.shape[0]
        new_n = old_n + 1

        # Assume _expand_square is in scope.
        sim_num = _expand_square(sim_num, new_n)
        sim_den = _expand_square(sim_den, new_n)
        b_sum = _expand_square(b_sum, new_n)
        b_cnt = _expand_square(b_cnt, new_n)

        members[new_c] = members[c1] + members[c2]

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

        sim_num[new_c, new_c] = 0.0
        sim_den[new_c, new_c] = 0.0
        b_sum[new_c, new_c] = 0.0
        b_cnt[new_c, new_c] = 0.0

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

        for bi in members[new_c]:
            domain_to_cluster[idx2dom[bi]] = new_c

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

        active = [x for x in active if x not in (c1, c2)]
        active.append(new_c)

        for old in (c1, c2):
            cluster_size_sum.pop(old, None)
            cluster_dti.pop(old, None)
            dti_weight_sum.pop(old, None)

    # ---------- 5. rename and write back ----------
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
    adata.obs[new_domain_key] = adata.obs[new_domain_key].cat.rename_categories(str)

    adata.uns[new_domain_key + "_map"] = meta_map
    adata.uns[new_domain_key + "_gain_log"] = pd.DataFrame(gain_log)

    if recompute_post:
        # Assume compute_transition_scores_entropy and compute_domain_sim are in scope.
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
    """Grow a square matrix in-place by padding with zeros to ``(new_n, new_n)``."""
    old_n = A.shape[0]
    if new_n <= old_n:
        return A
    B = np.zeros((new_n, new_n), dtype=A.dtype)
    B[:old_n, :old_n] = A
    return B
