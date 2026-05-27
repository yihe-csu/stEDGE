from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import seaborn as sns
from sklearn.decomposition import PCA


def prepare_interface_data(adata, belt_indices, gene_name):
    """Extract coordinates and gene expression for an interface belt.

    Parameters
    ----------
    adata : AnnData
    belt_indices : np.ndarray
        Indices of spots in the interface belt.
    gene_name : str
        Gene name to extract.

    Returns
    -------
    coords : np.ndarray
    expr : np.ndarray
    """
    coords = adata.obsm["spatial"][belt_indices]
    if hasattr(adata.raw.X, "toarray"):
        expr = adata[belt_indices, gene_name].X.toarray().flatten()
    else:
        expr = adata[belt_indices, gene_name].X.flatten()
    return coords, expr


def sort_points_by_main_axis(coords):
    """Sort points by their PCA projection onto the first principal axis.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (N, 2).

    Returns
    -------
    sorted_coords : np.ndarray
    sort_idx : np.ndarray
    """
    pca = PCA(n_components=1)
    projected_vals = pca.fit_transform(coords)
    sort_idx = np.argsort(projected_vals.flatten())
    return coords[sort_idx], sort_idx


def fit_interface_spline(sorted_coords, s=None, k=3):
    """Fit a parametric spline to sorted interface coordinates.

    Parameters
    ----------
    sorted_coords : np.ndarray
        Coordinates sorted along the main axis, shape (N, 2).
    s : float or None
        Smoothing parameter (larger = smoother).
    k : int
        Spline order (3 = cubic).

    Returns
    -------
    tck : tuple
        Spline representation.
    midline_points : np.ndarray
        Dense midline coordinates, shape (100, 2).
    """
    x = sorted_coords[:, 0]
    y = sorted_coords[:, 1]

    tck, _ = interpolate.splprep([x, y], s=s, k=k)

    u_new = np.linspace(0, 1, 100)
    midline_coords = interpolate.splev(u_new, tck)
    return tck, np.array(midline_coords).T


def plot_cross_boundary_gradient(distances, expr, gene_name, label_a, label_b, save_path=None):
    """Plot gene expression as a smoothed profile across a boundary interface.

    Parameters
    ----------
    distances : np.ndarray
        Signed distances from the interface midline (negative = side A).
    expr : np.ndarray
        Expression values for the gene.
    gene_name : str
        Gene name (used in the plot title).
    label_a : str
        Label for side A.
    label_b : str
        Label for side B.
    save_path : str or None
        If set, save the figure to this path.
    """
    df_plot = pd.DataFrame({'Distance': distances, 'Expression': expr})

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # light scatter of raw data
    sns.scatterplot(data=df_plot, x='Distance', y='Expression',
                    color='gray', s=6, alpha=0.1, edgecolor=None, ax=ax)

    # LOWESS smoothed cross-boundary profile
    sns.regplot(data=df_plot, x='Distance', y='Expression',
                scatter=False, lowess=True, color='#2878B5',
                line_kws={"linewidth": 2.5, "alpha": 0.9}, ax=ax)

    # interface centre line at x=0
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_title(f"$\\it{{{gene_name}}}$ Expression Profile", fontsize=11, pad=12)
    ax.set_ylabel("Normalized Expression", fontsize=9)
    ax.set_xlabel(
        f"Distance to Interface ($\\mu m$)\n"
        f" $\\longleftarrow$ {label_a} | {label_b} $\\longrightarrow$",
        fontsize=9,
    )

    sns.despine(trim=False)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_interface_skeleton(
    adata,
    tck,
    midline_points,
    side_a,
    side_b,
    label_a,
    label_b,
    save_path=None,
):
    """Debug visualisation of the interface spline skeleton.

    Shows the fitted midline, side-A and side-B spots, and spline
    start/end markers for verifying sort direction.

    Parameters
    ----------
    adata : AnnData
    tck : tuple
        Spline representation from ``fit_interface_spline``.
    midline_points : np.ndarray
    side_a, side_b : np.ndarray
        Boolean masks for each side of the interface.
    label_a, label_b : str
        Domain labels for annotation.
    save_path : str or None
    """
    n_obs = adata.n_obs
    mask_a = np.zeros(n_obs, dtype=bool)
    mask_b = np.zeros(n_obs, dtype=bool)

    idx_a = np.where(side_a)[0] if side_a.dtype == bool else side_a
    idx_b = np.where(side_b)[0] if side_b.dtype == bool else side_b
    mask_a[idx_a] = True
    mask_b[idx_b] = True

    coords_all = adata.obsm["spatial"]

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # background: all spots
    ax.scatter(coords_all[:, 0], coords_all[:, 1], c='lightgray', s=1, alpha=0.1)

    # side A (blue) and side B (orange)
    ax.scatter(coords_all[mask_a, 0], coords_all[mask_a, 1],
               c='#1f77b4', s=15, alpha=0.6, label=f'{label_a} (Dist < 0)')
    ax.scatter(coords_all[mask_b, 0], coords_all[mask_b, 1],
               c='#ff7f0e', s=15, alpha=0.6, label=f'{label_b} (Dist > 0)')

    # spline midline
    ax.plot(
        midline_points[:, 0],
        midline_points[:, 1],
        color='black',
        linewidth=2.5,
        linestyle='-',
        label='Interface Baseline (0)',
        zorder=10,
    )

    # spline start / end markers
    ax.scatter(
        midline_points[0, 0],
        midline_points[0, 1],
        c='green',
        s=50,
        marker='D',
        label='Spline Start',
        zorder=11,
    )
    ax.scatter(
        midline_points[-1, 0],
        midline_points[-1, 1],
        c='red',
        s=50,
        marker='X',
        label='Spline End',
        zorder=11,
    )

    ax.invert_yaxis()

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.set_aspect('equal')
    ax.set_title(f"Debug: {label_a} | {label_b} Interface Skeleton", fontsize=12)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y (Inverted)")

    pos_a = coords_all[mask_a].mean(axis=0)
    pos_b = coords_all[mask_b].mean(axis=0)
    ax.text(pos_a[0], pos_a[1], label_a, weight='bold', color='#1f77b4', ha='center')
    ax.text(pos_b[0], pos_b[1], label_b, weight='bold', color='#ff7f0e', ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def get_interface_skeleton(adata, side_a, side_b, seeds_a, seeds_b, s_para=30):
    """Fit a spline skeleton along the interface between two domains.

    Parameters
    ----------
    adata : AnnData
    side_a, side_b : np.ndarray
        Indices or boolean masks for spots on each side of the interface.
    seeds_a, seeds_b : np.ndarray
        Seed indices for each side.
    s_para : float
        Smoothing factor multiplier (larger = smoother spline).

    Returns
    -------
    tck : tuple
        Spline representation.
    midline_points : np.ndarray
        Dense midline coordinates.
    belt_indices_global : np.ndarray
        Global indices of spots in the interface belt.
    """
    def to_idx(s):
        return np.where(s)[0] if (isinstance(s, np.ndarray) and s.dtype == bool) else s

    idx_side_a = to_idx(side_a)
    idx_side_b = to_idx(side_b)
    idx_seeds_a = to_idx(seeds_a)
    idx_seeds_b = to_idx(seeds_b)

    belt_indices_global = np.union1d(idx_side_a, idx_side_b)
    fit_coords = adata.obsm["spatial"][belt_indices_global]

    sorted_coords, _ = sort_points_by_main_axis(fit_coords)
    tck, midline_points = fit_interface_spline(sorted_coords, s=len(fit_coords) * s_para)

    all_interface_indices = np.unique(np.concatenate([
        idx_side_a, idx_side_b, idx_seeds_a, idx_seeds_b
    ]))
    mask_debug = np.isin(np.arange(adata.n_obs), all_interface_indices)
    adata_debug = adata[mask_debug].copy()

    local_side_a = np.isin(all_interface_indices, idx_side_a)
    local_side_b = np.isin(all_interface_indices, idx_side_b)

    plot_interface_skeleton(
        adata_debug, tck, midline_points,
        side_a=local_side_a,
        side_b=local_side_b,
        label_a="Side A", label_b="Side B"
    )

    return tck, midline_points, belt_indices_global


def plot_gene_interface_gradient(
    adata,
    tck,
    side_a,
    side_b,
    seeds_a,
    seeds_b,
    gene_name,
    label_a,
    label_b,
    save_path=None,
):
    """Plot the cross-boundary expression profile of a gene along an interface."""
    all_interface_indices = np.unique(np.concatenate([
        np.where(side_a)[0] if side_a.dtype == bool else side_a,
        np.where(side_b)[0] if side_b.dtype == bool else side_b,
        np.where(seeds_a)[0] if seeds_a.dtype == bool else seeds_a,
        np.where(seeds_b)[0] if seeds_b.dtype == bool else seeds_b
    ]))

    mask_debug = np.isin(np.arange(adata.n_obs), all_interface_indices)
    adata_debug = adata[mask_debug].copy()

    idx_a_global = np.where(side_a)[0] if side_a.dtype == bool else side_a
    local_mask_a = np.isin(all_interface_indices, idx_a_global)

    coords = adata_debug.obsm["spatial"]

    if hasattr(adata_debug[:, gene_name].X, "toarray"):
        expr = adata_debug[:, gene_name].X.toarray().flatten()
    else:
        expr = adata_debug[:, gene_name].X.flatten()

    u_fine = np.linspace(0, 1, 2000)
    line_points = np.array(interpolate.splev(u_fine, tck)).T

    signed_distances = []
    for i, point in enumerate(coords):
        dists = np.linalg.norm(line_points - point, axis=1)
        min_dist = np.min(dists)
        val = -min_dist if local_mask_a[i] else min_dist
        signed_distances.append(val)

    signed_distances = np.array(signed_distances)

    plot_cross_boundary_gradient(
        signed_distances, expr, gene_name, label_a, label_b, save_path=save_path
    )

    return signed_distances, expr
