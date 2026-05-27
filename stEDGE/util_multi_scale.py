import time

import anndata as ad
import numpy as np
import scanpy as sc
import squidpy as sq
from scipy import sparse


def radius_representation(
    adata,
    use_rep="leiden",
    n_scales=4,
    nn_para=15,
    include_self=True,
    group_norm=False,
    dtype=np.float32,
):
    """
    Matrix-based multi-scale cell-type neighborhood representation.

    For each scale i, builds a spatial neighbour graph at radius
    ``nn_para * (i + 1)``, then computes the ring-like neighbourhood
    composition as::

        cur_X = (spatial_connectivities @ cell_type_one_hot) - prev_X

    Stores each scale in ``adata.obsm[f"scale{i}"]`` and calls
    ``generate_ct_representation_matrix`` to assemble a PCA-ready matrix.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs[use_rep]`` (categorical labels) and
        ``obsm["spatial"]``.
    use_rep : str
        Column in ``obs`` containing categorical cell/domain labels.
    n_scales : int
        Number of radius scales.
    nn_para : int or float
        Base radius. Radius at scale i is nn_para * (i + 1).
    include_self : bool
        Whether to include self-loop in the spatial neighbour graph.
    group_norm : bool
        Whether to normalise each row of the ring representation to sum to 1.
    dtype : numpy dtype
        Data type for computed matrices.
    """

    # Ensure categorical labels
    if not hasattr(adata.obs[use_rep], "cat"):
        adata.obs[use_rep] = adata.obs[use_rep].astype("category")

    cls = adata.obs[use_rep].cat
    labels = cls.codes.to_numpy()
    cell_types = np.array(cls.categories)

    n_cells = adata.n_obs
    n_types = len(cell_types)

    print(f">>> n_cells = {n_cells}")
    print(f">>> n_types = {n_types}")

    # Build one-hot matrix: cells × cell types
    valid = labels >= 0
    one_hot = sparse.csr_matrix(
        (
            np.ones(np.sum(valid), dtype=dtype),
            (np.where(valid)[0], labels[valid]),
        ),
        shape=(n_cells, n_types),
        dtype=dtype,
    )

    ME_X_prev = np.zeros((n_cells, n_types), dtype=dtype)

    for i in range(n_scales):
        cur_scale = i
        actual_r = nn_para * (cur_scale + 1)

        print(f"\nscale {cur_scale}, radius={actual_r}")
        t0 = time.time()

        # Build spatial graph
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            radius=actual_r,
            set_diag=include_self,
        )

        I = adata.obsp["spatial_connectivities"].tocsr().astype(dtype)

        # Key matrix operation:
        # cell × cell adjacency  @  cell × cell-type one-hot
        # = cell × cell-type neighborhood count matrix
        ME_X = I @ one_hot

        # Convert to dense because downstream code expects dense obsm arrays
        if sparse.issparse(ME_X):
            ME_X = ME_X.toarray()

        ME_X = ME_X.astype(dtype, copy=False)

        # Ring-like representation:
        # current radius neighborhood minus previous radius neighborhood
        cur_X = ME_X - ME_X_prev
        ME_X_prev = ME_X

        print(
            f"scale {cur_scale}, median #cells per radius shell:",
            np.median(np.sum(cur_X, axis=1)),
        )
        print(f">>> matrix computation completed in {time.time() - t0:.2f} seconds")

        cur_ME_key = f"scale{cur_scale}"

        if group_norm:
            denom = np.sum(cur_X, axis=1, keepdims=True)
            cur_X = np.divide(
                cur_X,
                denom,
                out=np.zeros_like(cur_X, dtype=dtype),
                where=denom != 0,
            )

        adata.obsm[cur_ME_key] = cur_X.copy()

    generate_ct_representation_matrix(
        adata,
        use_rep=use_rep,
        n_scales=n_scales,
    )


def generate_ct_representation_matrix(adata, use_rep="leiden", n_scales=4):
    """Assemble a multi-scale cell-type representation matrix and compute PCA.

    Concatenates per-scale, per-cell-type neighbourhood composition vectors
    into a single feature matrix ``adata.obsm["whole"]``, builds a temporary
    AnnData, and runs normalisation → log1p → PCA.  The resulting PCs are
    stored in ``adata.obsm["X_pca"]``.

    This is called automatically by ``radius_representation``.

    Parameters
    ----------
    adata : AnnData
        Must contain ``obs[use_rep]`` and ``obsm[f"scale{i}"]`` for each scale.
    use_rep : str
        Column in ``obs`` containing categorical labels.
    n_scales : int
        Number of radius scales (must match ``radius_representation``).
    """
    if not hasattr(adata.obs[use_rep], "cat"):
        adata.obs[use_rep] = adata.obs[use_rep].astype("category")

    cell_types = np.array(adata.obs[use_rep].cat.categories)

    whole_feature_list = []
    whole_feature_X = []

    for ct_idx in range(len(cell_types)):
        rep_list = []

        for i in range(n_scales):
            x = adata.obsm[f"scale{i}"][:, ct_idx]
            rep_list.append(x)
            whole_feature_list.append(f"ct{ct_idx}scale{i}")
            whole_feature_X.append(x)

        cur_ct_rep = np.array(rep_list).T
        adata.obsm[f"ct{ct_idx}"] = cur_ct_rep

    whole_X = np.array(whole_feature_X).T
    adata.obsm["whole"] = whole_X

    adata_feature = ad.AnnData(X=whole_X)
    adata_feature.obs_names = adata.obs_names
    adata_feature.var_names = whole_feature_list
    adata_feature.obsm["spatial"] = adata.obsm["spatial"].copy()

    for k in adata.obs.keys():
        adata_feature.obs[k] = adata.obs[k].copy()

    if "spatial" in adata.uns:
        adata_feature.uns["spatial"] = adata.uns["spatial"]

    sc.pp.normalize_total(adata_feature)
    sc.pp.log1p(adata_feature)
    sc.pp.pca(adata_feature)

    adata.obsm["X_pca"] = adata_feature.obsm["X_pca"].copy()

    print(">>> adata.obsm['whole'] generated!")
    print(">>> adata.obsm['X_pca'] generated from adata.obsm['whole']!")
