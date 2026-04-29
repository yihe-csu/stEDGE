import time
import numpy as np
import squidpy as sq
import anndata as ad
import scanpy as sc
from joblib import Parallel, delayed


def _compute_neighbors_block(start, end, indptr, indices, labels, n_types):
    block = np.zeros((end - start, n_types), dtype=np.int32)
    for row_offset, cell_idx in enumerate(range(start, end)):
        cur_neighbors = indices[indptr[cell_idx] : indptr[cell_idx + 1]]
        block[row_offset, :] = np.bincount(labels[cur_neighbors], minlength=n_types)
    return start, block


def radius_representation(
    adata,
    use_rep="leiden",
    n_scales=4,
    nn_para=15,
    include_self=True,
    group_norm=False,
    n_jobs=-1,
    block_size=4096,
):
    cls_array = adata.obs[use_rep]
    ME_var_names_np_unique = np.array(adata.obs[use_rep].cat.categories)
    labels = cls_array.cat.codes.to_numpy()

    ME_X_prev = np.zeros(shape=(cls_array.shape[0], ME_var_names_np_unique.shape[0]))

    for i in range(n_scales):
        cur_scale = i
        print(f"scale {cur_scale}")

        sq.gr.spatial_neighbors(
            adata, coord_type="generic", radius=nn_para * (cur_scale + 1), set_diag=include_self
        )
        I = adata.obsp["spatial_connectivities"].tocsr()
        indptr = I.indptr
        indices = I.indices
        n_cells = I.shape[0]
        n_types = ME_var_names_np_unique.shape[0]
        ME_X = np.zeros(shape=(n_cells, n_types), dtype=np.int32)

        if n_jobs == 1:
            starts = list(range(0, n_cells, block_size))
            for start in starts:
                end = min(start + block_size, n_cells)
                _, block = _compute_neighbors_block(start, end, indptr, indices, labels, n_types)
                ME_X[start:end, :] = block
        else:
            starts = list(range(0, n_cells, block_size))
            print(
                f">>> Parallel neighbor computation: n_jobs={n_jobs}, "
                f"blocks={len(starts)}, block_size={block_size}"
            )
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1)(
                delayed(_compute_neighbors_block)(
                    start, min(start + block_size, n_cells), indptr, indices, labels, n_types
                )
                for start in starts
            )
            for start, block in results:
                end = start + block.shape[0]
                ME_X[start:end, :] = block
            print(f">>> Neighbor computation completed in {time.time() - start_time:.2f} seconds")
        cur_ME_key = f"scale{cur_scale}"

        cur_X = ME_X - ME_X_prev
        ME_X_prev = ME_X

        actual_r = nn_para * (cur_scale + 1)
        print(
            f"scale {cur_scale}, median #cells per radius (r={actual_r}):",
            np.median(np.sum(cur_X, axis=1)),
        )

        adata.obsm[cur_ME_key] = cur_X.copy()
        if group_norm:
            adata.obsm[cur_ME_key] = adata.obsm[cur_ME_key] / np.sum(
                adata.obsm[cur_ME_key], axis=1, keepdims=True
            )
            adata.obsm[cur_ME_key] = np.nan_to_num(adata.obsm[cur_ME_key], 0)
    generate_ct_representation(adata, use_rep=use_rep, n_scales=n_scales)


def generate_ct_representation(adata, use_rep="leiden", n_scales=4):
    
    ME_var_names_np_unique = np.array(adata.obs[use_rep].cat.categories)
    whole_feature_list = []
    while_feature_X = []
    for ct_idx in range(len(ME_var_names_np_unique)):
        rep_list = []
        for i in range(n_scales):
            rep_list.append(adata.obsm[f'scale{i}'][:,ct_idx])
            whole_feature_list.append(f'ct{ct_idx}scale{i}')
            while_feature_X.append(adata.obsm[f'scale{i}'][:,ct_idx])

        cur_ct_rep = np.array(rep_list).transpose()
        cur_obsm = f'ct{ct_idx}'
        adata.obsm[cur_obsm] = cur_ct_rep

    adata.obsm[f'whole'] = np.array(while_feature_X).transpose()

    # make another anadata for whole feature
    adata_feature = ad.AnnData(X = np.array(while_feature_X).transpose())
    adata_feature.obs_names = adata.obs_names
    adata_feature.var_names = whole_feature_list
    adata_feature.obsm['spatial'] = adata.obsm['spatial']
    # adata_obs_keys = self.adata.obs.keys()
    for k in adata.obs.keys():
        adata_feature.obs[k] = adata.obs[k]
    # for k in self.adata.uns.keys()
    if 'spatial' in adata.uns:
        adata_feature.uns['spatial'] = adata.uns['spatial']
    # adata.obsm['scale_feat'] = adata_feature

    sc.pp.normalize_total(adata_feature)
    sc.pp.log1p(adata_feature)
    sc.pp.pca(adata_feature)
    sc.pp.neighbors(adata_feature)
    sc.tl.leiden(adata_feature, key_added="leiden_scale", resolution=1)
    adata.obsm["X_pca"] = adata_feature.obsm["X_pca"].copy()
    adata.obs["leiden_scale"] = adata_feature.obs["leiden_scale"].copy()
    print(">>> adata.obsm['whole'] generated !")
    print(">>> adata.obsm['X_pca'] generated (use_rep=adata.obsm['whole'])!")
    print(">>> adata.obsm['leiden_scale'] generated!")
