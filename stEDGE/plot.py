import os
from pathlib import Path
from typing import Optional, Tuple, Union

import anndata as ad
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import stEDGE
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import davies_bouldin_score

from . import TGL_module


# =============================================================================
# Shared plotting utilities
# =============================================================================

FigSize = Tuple[float, float]
PathLike = Union[str, os.PathLike]


def _ensure_output_dir(output_dir: Optional[PathLike]) -> Optional[Path]:
    """Create and return an output directory path, or None if not requested."""
    if output_dir is None:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_filename(value) -> str:
    """Return a filesystem-safe string for use in generated filenames."""
    s = str(value)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        s = s.replace(ch, '_')
    return s


def _resolve_save_path(
    output_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> Optional[Path]:
    """Resolve either an explicit save path or an output_dir/filename pair."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if output_dir is None or filename is None:
        return None
    return _ensure_output_dir(output_dir) / filename


def _save_show_close(
    fig,
    output_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    dpi: int = 600,
    show: bool = False,
    close: Optional[bool] = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
):
    """Save, display, and close a Matplotlib figure in a consistent way."""
    resolved = _resolve_save_path(output_dir=output_dir, filename=filename, save_path=save_path)

    if resolved is not None:
        fig.savefig(resolved, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

    if show:
        plt.show()

    if close is None:
        close = not show
    if close:
        plt.close(fig)

    return resolved


def _draw_sankey_ribbon(ax, p1, p2, color, alpha=0.5, width_px=8, n=30,
                        zorder=1, outline=False, outline_lw=0.4):
    """Draw a Bezier ribbon between two points with fixed pixel width.

    Used by ``plot_domain_tree`` and ``plot_domain_tree_spatial`` to render
    variable-width edges whose thickness reflects DTI intensity.
    """
    x1, y1 = p1
    x2, y2 = p2
    midy = (y1 + y2) / 2.0

    P0 = np.array([x1, y1], float)
    P1 = np.array([x1, midy], float)
    P2 = np.array([x2, midy], float)
    P3 = np.array([x2, y2], float)

    t = np.linspace(0, 1, n)
    B = ((1 - t) ** 3)[:, None] * P0 \
        + (3 * (1 - t) ** 2 * t)[:, None] * P1 \
        + (3 * (1 - t) * t ** 2)[:, None] * P2 \
        + (t ** 3)[:, None] * P3

    B_disp = ax.transData.transform(B)
    d = np.gradient(B_disp, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    nrm = nrm / nrm_len

    off = (width_px / 2.0) * nrm
    top_disp = B_disp + off
    bot_disp = B_disp - off

    inv = ax.transData.inverted()
    top = inv.transform(top_disp)
    bot = inv.transform(bot_disp)

    poly = np.vstack([top, bot[::-1], top[0:1]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(poly) - 2) + [mpath.Path.CLOSEPOLY]
    path = mpath.Path(poly, codes)

    patch = mpatches.PathPatch(
        path, facecolor=color,
        edgecolor=(color if outline else "none"),
        lw=(outline_lw if outline else 0),
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def plot_cluster_count_resolution(
    adata,
    clusters_key="clusters_results",
    figsize=(5, 2),
    color="#2166AC",
    marker_size=7,
    linewidth=1.5,
    title="Cluster count vs. resolution",
    output_dir=None,
    filename="cluster_count_vs_resolution.png",  # Recommended PDF output
    show=True,
    close=None,
    dpi=600,
    ax=None,     # Supports external axes for composite figures
):
    """
    Plot the number of clusters across Leiden resolutions.
    Nature-style formatting applied (editable fonts, no DPI locks).
    """
    if clusters_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing key: '{clusters_key}'")

    clusters_df = pd.DataFrame(adata.obsm[clusters_key])

    def _parse_resolution(col):
        try:
            return float(str(col).split("_")[1])
        except Exception as e:
            raise ValueError(
                f"Cannot parse resolution from column '{col}'. "
                "Expected format like 'leiden_0.1'."
            ) from e

    resolutions = np.array([_parse_resolution(c) for c in clusters_df.columns])
    n_clusters = np.array([clusters_df[c].nunique() for c in clusters_df.columns])

    order = np.argsort(resolutions)
    resolutions = resolutions[order]
    n_clusters = n_clusters[order]

    # ==========================================
    # Avoid forcing display DPI and keep editable PDF fonts.
    # ==========================================
    style = {
        # "figure.dpi": dpi,  # Do not override native Jupyter rendering.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "pdf.fonttype": 42,   # Keep fonts editable in Illustrator
        "ps.fonttype": 42
    }

    with mpl.rc_context(style):
        # ==========================================
        # Support externally provided axes.
        # ==========================================
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.plot(
            resolutions,
            n_clusters,
            color=color,
            linewidth=linewidth,
            linestyle="-",
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=2,
        )

        ax.set_xlabel("Leiden resolution", weight="medium")
        ax.set_ylabel("Number of clusters", weight="medium")
        ax.set_title(title, fontsize=11, color="#333333", pad=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=5,
        )

        ax.grid(False)

        # ==========================================
        # Only standalone figures handle layout and saving.
        # ==========================================
        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(
                    fig,
                    output_dir=output_dir,
                    filename=filename,
                    dpi=dpi,
                    show=show,
                    close=close,
                )
            elif show:
                plt.show()

    return fig, ax, saved_path


def plot_spatial(
    adata,
    colors,
    titles=None,
    spot_size=10,
    cmap="inferno",
    palette=None,
    figsize=(5, 5),
    alpha=1.0,
    frameon=False,
    legend_loc=None,
    colorbar_loc="right",
    output_dir=None,
    filename=None,
    save_path=None,
    show=False,
    close=None,
    dpi=600,
    ax=None,
    base_fontsize=None,  # Base font size; auto-scaled when None
):
    """Reusable ``scanpy.pl.spatial`` wrapper with dynamic font scaling and Nature-style formatting."""

    if isinstance(colors, str):
        colors = [colors]
    if titles is None:
        titles = colors
    if filename is None and output_dir is not None:
        filename = f"spatial_{'_'.join(map(str, colors))}.png"

    # ==========================================
    # Dynamic font-size scaling.
    # ==========================================
    if base_fontsize is None:
        # Estimate font size from figure height.
        # Scale empirically and clamp to a publication-friendly range.
        base_fontsize = max(7, min(14, int(figsize[1] * 2.2)))

    nature_rc = {
        "figure.figsize": figsize,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # Font sizes are relative to base_fontsize.
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,    # Title is 1 pt larger
        "legend.fontsize": base_fontsize - 1,   # Legend is 1 pt smaller
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }

    with plt.rc_context(nature_rc):
        is_standalone = (ax is None)

        sc.pl.spatial(
            adata,
            img_key=None,
            color=colors,
            cmap=cmap,
            palette=palette,
            title=titles,
            alpha=alpha,
            frameon=frameon,
            legend_loc=legend_loc,
            colorbar_loc=colorbar_loc,
            spot_size=spot_size,
            show=False,
            ax=ax,
        )

        if is_standalone:
            fig = plt.gcf()
        else:
            fig = ax.get_figure()
        axes = fig.axes

    saved_path = None
    if is_standalone:
        if '_save_show_close' in globals():
            saved_path = _save_show_close(
                fig, output_dir=output_dir, filename=filename,
                save_path=save_path, dpi=dpi, show=show, close=close,
            )
        elif show:
            plt.show()

    return fig, axes, saved_path


def plot_p_edge_distribution(
    adata,
    bins=50,
    color="#4C72B0",
    show_median=False,
    seed_quantile=None,
    density=False,
    figsize=(5, 2.6),
    output_dir=None,
    filename="p_edge_distribution.png",  # Recommended vector output
    show=False,
    close=None,
    dpi=600,
    ax=None,             # Supports external axes
):
    """Plot the distribution of boundary probability ``p_edge``."""
    if "p_edge" not in adata.obs:
        raise KeyError("adata.obs missing 'p_edge'.")

    p_edge = np.asarray(adata.obs["p_edge"].values, dtype=float)
    p_edge = p_edge[np.isfinite(p_edge)]

    style = {
        # Keep native rendering by not setting figure.dpi.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,       # Editable PDF fonts
        "ps.fonttype": 42         # Editable PS fonts
    }

    with plt.rc_context(style):
        # Use the external axes.
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.hist(
            p_edge,
            bins=np.linspace(0, 1, bins + 1),
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.95,
            density=density,
        )

        if seed_quantile is not None:
            p_cap = float(np.quantile(p_edge, seed_quantile))
            ax.axvline(p_cap, color="#D62728", linestyle="--", linewidth=1.2, zorder=4)
            ax.axvspan(0, p_cap, color="#D62728", alpha=0.15, zorder=1)
            ax.text(
                p_cap + 0.02, ax.get_ylim()[1] * 0.9,
                f"Seed cutoff\n(top {seed_quantile:.0%}, p={p_cap:.2f})",
                ha="left", va="top", fontsize=7, color="#D62728", zorder=5
            )

        if show_median:
            median = float(np.median(p_edge))
            ax.axvline(
                median,
                ymin=0,
                ymax=0.65,
                color="black",
                linestyle="--",
                linewidth=1.0,
                zorder=4,
            )
            ax.text(
                median * 0.94 if median > 0.5 else median * 1.06,
                ax.get_ylim()[1] * 0.65,
                f"Median = {median:.3f}",
                ha="right" if median > 0.5 else "left",
                va="top",
                fontsize=8,
                color="black",
                rotation=90,
                zorder=5,
            )

        ax.set_xlabel(r"Boundary probability ($p_{\mathrm{edge}}$)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_xlim(0, 1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="out")

        if not density:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Adjust layout and save only for standalone figures.
        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(
                    fig, output_dir=output_dir, filename=filename,
                    dpi=dpi, show=show, close=close,
                )
            elif show:
                plt.show()

    return fig, ax, saved_path


def plot_gradient_field(
    adata,
    step=1,
    scale=150,
    flip_y=True,
    show_points=True,
    point_alpha=0.3,
    point_size=2,        # Keep background points small
    cmap="RdBu_r",
    figsize=(3.5, 3.5),  # Compact default figure size
    output_dir=None,
    filename="gradient_field.png",  # Recommended vector output
    show=False,
    close=None,
    dpi=600,
    ax=None,
):
    """Plot the spatial gradient field of ``p_edge`` with Nature-style formatting."""
    for key in ["spatial", "grad_p_edge"]:
        if key not in adata.obsm:
            raise KeyError(f"adata.obsm missing '{key}'.")
    if "p_edge" not in adata.obs:
        raise KeyError("adata.obs missing 'p_edge'.")

    coords = np.asarray(adata.obsm["spatial"])
    p_edge = np.asarray(adata.obs["p_edge"].values)
    grad_p_edge = np.asarray(adata.obsm["grad_p_edge"])
    idx = np.arange(0, len(coords), max(1, int(step)))

    x = coords[:, 0]
    y = -coords[:, 1] if flip_y else coords[:, 1]
    gx = grad_p_edge[:, 0]
    gy = -grad_p_edge[:, 1] if flip_y else grad_p_edge[:, 1]
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gx_norm = gx / (grad_mag + 1e-9)
    gy_norm = gy / (grad_mag + 1e-9)

    # ==========================================
    # Local Nature-style rendering context.
    # Limit preview DPI to avoid oversized notebook output.
    # ==========================================
    style = {
        "figure.figsize": figsize,
        "figure.dpi": 100,             # Limit Jupyter preview size
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,            # Keep PDF fonts editable
        "ps.fonttype": 42
    }

    with plt.rc_context(style):
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(100)  # Limit preview size
        else:
            fig = ax.get_figure()

        if show_points:
            # Draw the background point cloud with a simple gray map.
            ax.scatter(x, y, c=p_edge, cmap="gray", s=point_size, alpha=point_alpha, linewidths=0, zorder=1)

        # Draw gradient vectors.
        # Use thinner arrow heads and shafts for a sharper look.
        q = ax.quiver(
            x[idx], y[idx], gx_norm[idx], gy_norm[idx], grad_mag[idx],
            cmap=cmap, scale=scale, headwidth=4, width=0.0025, zorder=2
        )

        # Refine colorbar styling.
        # Keep the colorbar slim and close to the main plot.
        cbar = fig.colorbar(q, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Gradient magnitude", weight='medium')
        cbar.outline.set_linewidth(0.5)  # Thin colorbar border
        cbar.ax.tick_params(length=2, width=0.5)  # Thin ticks

        ax.set_title("Gradient field", pad=10)  # Optional notebook title
        ax.set_aspect("equal")
        ax.axis("off")  # Hide axes frame

        fig.tight_layout()

        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
            elif show:
                plt.show()

    return fig, ax, saved_path


def plot_gradient_magnitude(
    adata,
    flip_y=True,
    point_size=2,        # Reduce point size for compact figures
    figsize=(3.5, 3.5),  # Compact single-column figure size
    cmap="RdBu_r",      # Keep this colormap for magnitude values
    output_dir=None,
    filename="gradient_magnitude.pdf",  # Default vector PDF output
    show=False,
    close=None,
    dpi=600,
    ax=None,
):
    """Plot the magnitude of ``grad_p_edge`` as a spatial scatter plot with Nature-style formatting."""
    for key in ["spatial", "grad_p_edge"]:
        if key not in adata.obsm:
            raise KeyError(f"adata.obsm missing '{key}'.")

    coords = np.asarray(adata.obsm["spatial"])
    grad_magnitude = np.linalg.norm(np.asarray(adata.obsm["grad_p_edge"]), axis=1)
    x = coords[:, 0]
    y = -coords[:, 1] if flip_y else coords[:, 1]

    # ==========================================
    # Local Nature-style rendering context.
    # Limit preview DPI to avoid oversized notebook output.
    # ==========================================
    style = {
        "figure.figsize": figsize,
        "figure.dpi": 100,             # Limit Jupyter preview size
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,          # Slightly larger title
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,            # Keep PDF fonts editable
        "ps.fonttype": 42
    }

    with plt.rc_context(style):
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(100)  # Limit preview size
        else:
            fig = ax.get_figure()

        # Draw scatter plot.
        sca = ax.scatter(x, y, c=grad_magnitude, cmap=cmap, s=point_size, linewidths=0)

        # ==========================================
        # Keep the colorbar slim and close to the main plot.
        # ==========================================
        cbar = fig.colorbar(sca, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Gradient magnitude", weight='medium')
        cbar.outline.set_linewidth(0.5)          # Thin border
        cbar.ax.tick_params(length=2, width=0.5) # Thin ticks

        # ==========================================
        # Refine title and axes.
        # ==========================================
        ax.set_title("Gradient Magnitude", pad=12, weight='medium')
        ax.set_aspect("equal")
        ax.axis("off")  # Hide axes frame

        fig.tight_layout()


        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
            elif show:
                plt.show()


    return fig, ax, saved_path


def plot_umap_with_db_index(
    adata,
    label_key="domain_merge",
    umap_key="X_umap",
    figsize=(5, 5),
    s=30,
    outline_width=(0.1, 0.05),
    legend_fontsize=10,
    output_dir=None,
    filename="stedge_umap.png",
    output_path=None,
    show=False,
    close=None,
    dpi=600,
):
    """Plot UMAP coloured by a label and report the Davies-Bouldin index."""
    if umap_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing '{umap_key}'.")
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs missing '{label_key}'.")

    X = np.asarray(adata.obsm[umap_key])
    labels = adata.obs[label_key].values
    db_index = davies_bouldin_score(X, labels)

    with plt.rc_context({"figure.figsize": figsize}):
        sc.pl.umap(
            adata,
            color=label_key,
            title=[f"UMAP colored by {label_key} (DB: {db_index:.3f})"],
            legend_fontoutline=1,
            add_outline=True,
            s=s,
            outline_width=outline_width,
            legend_fontsize=legend_fontsize,
            frameon=False,
            show=False,
        )
        fig = plt.gcf()
        axes = fig.axes

    saved_path = _save_show_close(
        fig,
        output_dir=output_dir,
        filename=filename,
        save_path=output_path,
        dpi=dpi,
        show=show,
        close=close,
    )
    return fig, axes, saved_path


def plot_top_genes_per_domain_combined(
    adata,
    group_key="domain",
    top_n=5,
    spot_size=350,
    panel_width=3,
    panel_height=4,
    output_dir=None,
    out_dir=None,
    filename_prefix="domain",
    show=False,
    close=None,
    dpi=600,
):
    """Plot each domain and its top marker genes as a combined spatial panel."""
    output_dir = output_dir if output_dir is not None else out_dir
    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    if "rank_genes_groups" not in adata.uns:
        raise KeyError("adata.uns missing 'rank_genes_groups'. Run sc.tl.rank_genes_groups first.")

    output_dir = _ensure_output_dir(output_dir)
    markers = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_groups")
    top_markers = markers.groupby("group", observed=True).head(top_n)
    saved_paths = []

    for domain in top_markers["group"].unique():
        genes = top_markers[top_markers["group"] == domain]["names"].tolist()
        domain_str = _safe_filename(domain)
        fig, axes = plt.subplots(1, len(genes) + 1, figsize=(panel_width * (len(genes) + 1), panel_height))
        axes = np.asarray(axes).ravel()

        sc.pl.spatial(
            adata,
            color=group_key,
            groups=[domain],
            spot_size=spot_size,
            ax=axes[0],
            show=False,
            title=f"Domain {domain}",
        )
        for i, gene in enumerate(genes, start=1):
            sc.pl.spatial(
                adata,
                color=gene,
                spot_size=spot_size,
                ax=axes[i],
                show=False,
                title=gene,
            )

        fig.tight_layout()
        saved_path = _save_show_close(
            fig,
            output_dir=output_dir,
            filename=f"{filename_prefix}_{domain_str}_combined.png",
            dpi=dpi,
            show=show,
            close=close,
        )
        saved_paths.append(saved_path)

    return saved_paths


def plot_domain_with_alpha_series(
    adata,
    color_key="domain",
    alphas=None,
    cmap="inferno",
    spot_size=350,
    figsize=(10, 10),
    output_dir=None,
    filename_prefix=None,
    show=False,
    close=None,
    dpi=600,
):
    """Plot a spatial categorical/continuous field over a series of alpha values."""
    if color_key not in adata.obs and color_key not in adata.var_names:
        raise KeyError(f"'{color_key}' not found in adata.obs or adata.var_names.")
    if output_dir is None:
        raise ValueError("output_dir must be provided for alpha series export.")
    if alphas is None:
        alphas = np.arange(0, 1.01, 0.1)
    if filename_prefix is None:
        filename_prefix = _safe_filename(color_key)

    saved_paths = []
    for alpha in alphas:
        with plt.rc_context({"figure.figsize": figsize}):
            sc.pl.spatial(
                adata,
                color=color_key,
                cmap=cmap,
                title=[f"alpha = {alpha:.1f}"],
                alpha=float(alpha),
                frameon=False,
                colorbar_loc=None,
                spot_size=spot_size,
                show=False,
            )
            fig = plt.gcf()
        saved_path = _save_show_close(
            fig,
            output_dir=output_dir,
            filename=f"{filename_prefix}_alpha_{alpha:.1f}.png",
            dpi=dpi,
            show=show,
            close=close,
        )
        saved_paths.append(saved_path)
    return saved_paths


def plot_clusters_results_spatial(
    adata,
    spot_size=250,
    figsize=(8, 8),
    output_dir=None,
    filename_prefix="cluster",
    show=False,
    close=None,
    dpi=600,
    cleanup=True,
):
    """Visualise each column of ``adata.obsm['clusters_results']`` spatially."""
    if "clusters_results" not in adata.obsm:
        raise KeyError("adata.obsm missing 'clusters_results'.")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm missing 'spatial'.")
    if output_dir is None:
        raise ValueError("output_dir must be provided.")

    clusters_results = pd.DataFrame(adata.obsm["clusters_results"], index=adata.obs_names)
    saved_paths = []
    tmp_cols = []

    for col in clusters_results.columns:
        tmp_col = f"_tmp_cluster_{_safe_filename(col)}"
        tmp_cols.append(tmp_col)
        adata.obs[tmp_col] = clusters_results[col].astype("category")

        with plt.rc_context({"figure.figsize": figsize}):
            sc.pl.spatial(
                adata,
                color=tmp_col,
                cmap="inferno",
                title=[f"Cluster {col}"],
                alpha=1,
                frameon=False,
                colorbar_loc=None,
                spot_size=spot_size,
                show=False,
            )
            fig = plt.gcf()

        saved_path = _save_show_close(
            fig,
            output_dir=output_dir,
            filename=f"{filename_prefix}_{_safe_filename(col)}.png",
            dpi=dpi,
            show=show,
            close=close,
        )
        saved_paths.append(saved_path)

    if cleanup:
        adata.obs.drop(columns=[c for c in tmp_cols if c in adata.obs], inplace=True)

    return saved_paths


def batch_detect_and_plot(
    adata,
    output_dir,
    spot_size=20,
    min_size=20,
    th_range=np.arange(0.1, 0.91, 0.1),
    con_region_key="con_region_ori",
    figsize=(8, 8),
    show=False,
    close=None,
    dpi=600,
):
    """Batch detect new domains over thresholds and export spatial plots.

    Note
    ----
    This function modifies ``adata`` by calling
    ``stEDGE.utils.detect_new_domains_from_unassigned``.  It is retained for
    backward compatibility; new analysis code should keep detection outside the
    plotting module when possible.
    """
    out_dir = _ensure_output_dir(Path(output_dir) / "th_inner")
    saved_paths = []

    for th_inner in th_range:
        stEDGE.utils.detect_new_domains_from_unassigned(
            adata,
            th_inner=float(th_inner),
            min_size=min_size,
            con_region_key=con_region_key,
        )
        with plt.rc_context({"figure.figsize": figsize}):
            sc.pl.spatial(
                adata,
                color="con_region",
                cmap="inferno",
                title=[f"th_inner_{th_inner:.1f}"],
                alpha=1,
                frameon=False,
                colorbar_loc=None,
                spot_size=spot_size,
                show=False,
            )
            fig = plt.gcf()
        saved_path = _save_show_close(
            fig,
            output_dir=out_dir,
            filename=f"con_region_{th_inner:.1f}.png",
            dpi=dpi,
            show=show,
            close=close,
        )
        saved_paths.append(saved_path)

    return saved_paths


def plot_domain_graph(
    adata,
    domain_key: str = "domain",
    color_mode: str = "dti",
    min_sim_threshold: float = None,  # Default to None
    min_sim_quantile: float = 0.65,   # Default to 0.65
    max_edge_width: float = 8.0,
    node_size_base: float = 80,
    node_size_scale: float = 100,
    figsize=(8, 6),
    legend=False,
    output_dir=None,
    filename=None,
    show: bool = False,
    close=None,
    dpi: int = 600,
    ax=None,
    return_graph: bool = False,
):
    """Draw a domain similarity graph with consistent save/show handling and subplot support."""
    color_mode = color_mode.lower()
    if color_mode not in {"dti", "cluster"}:
        raise ValueError("color_mode must be 'dti' or 'cluster'.")
    if filename is None:
        filename = f"{domain_key}_graph_{color_mode}.png"

    similarity_key = f"{domain_key}_similarity"
    if similarity_key not in adata.uns:
        raise KeyError(
            f"{similarity_key} not found in adata.uns; "
            "build the domain similarity matrix first."
        )

    sim_df = adata.uns[similarity_key].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)
    domains = sim_df.index.to_list()
    K = len(domains)
    if K == 0:
        print("No domains to plot.")
        return (None, None, None, None) if return_graph else (None, None, None)

    dom_series = adata.obs[domain_key].astype(str)
    size_counts = dom_series.value_counts()
    domain_sizes = np.array([size_counts.get(d, 0) for d in domains], dtype=float)
    if domain_sizes.max() > domain_sizes.min():
        size_norm = (domain_sizes - domain_sizes.min()) / (domain_sizes.max() - domain_sizes.min())
    else:
        size_norm = np.zeros_like(domain_sizes)
    node_sizes = node_size_base + node_size_scale * size_norm

    dti_series = None
    dti_key = f"{domain_key}_DTI"
    if dti_key in adata.obs.columns:
        dti_series = adata.obs.groupby(domain_key, observed=True)[dti_key].first()
        dti_series = dti_series.copy()
        dti_series.index = dti_series.index.astype(str)

    cluster_color_map = None
    dom2cl = None
    cluster_cats = None
    if color_mode == "cluster":
        cluster_key = f"{domain_key}_coarse"
        coarse_color_key = f"{domain_key}_coarse_colors"
        if cluster_key not in adata.obs.columns:
            raise KeyError(f"color_mode='cluster' but column '{cluster_key}' not found in adata.obs.")
        cluster_series = adata.obs[cluster_key].astype("category")
        cluster_cats = cluster_series.cat.categories.astype(str)
        if coarse_color_key in adata.uns:
            palette = list(adata.uns[coarse_color_key])
        else:
            cmap_tab = plt.colormaps.get_cmap("tab20")
            palette = [mcolors.to_hex(cmap_tab(i % cmap_tab.N)) for i in range(len(cluster_cats))]
            adata.uns[coarse_color_key] = palette
        if len(palette) < len(cluster_cats):
            palette = (palette * (len(cluster_cats) // len(palette) + 1))[: len(cluster_cats)]
        cluster_color_map = dict(zip(cluster_cats, palette))
        dom2cl = (
            adata.obs[[domain_key, cluster_key]]
            .astype({domain_key: str})
            .groupby(domain_key, observed=True)[cluster_key]
            .first()
        )
        dom2cl.index = dom2cl.index.astype(str)

    G = nx.Graph()
    for d in domains:
        d_str = str(d)
        dti_val = float(dti_series.get(d_str)) if dti_series is not None and d_str in dti_series.index else np.nan
        cl_val = dom2cl.get(d_str) if dom2cl is not None and d_str in dom2cl.index else np.nan
        G.add_node(d_str, DTI=dti_val, CLUSTER=cl_val)

    sim = sim_df.to_numpy(dtype=float)
    upper = sim[np.triu_indices(K, k=1)]
    if upper.size == 0:
        print("No domain pairs to connect.")
        return (None, None, None, G) if return_graph else (None, None, None)

    # ==========================================
    # Prefer absolute thresholds to match clustering behavior.
    # ==========================================
    if min_sim_threshold is not None:
        thr = float(min_sim_threshold)
        mode = "absolute"
    else:
        thr = 0.0
        if min_sim_quantile is not None:
            thr = float(np.quantile(upper, min_sim_quantile))
        mode = f"quantile({min_sim_quantile})"

    # Optional log showing the threshold used for plotting.
    # print(f">>> plot_domain_graph edge threshold set to {thr:.4f} via {mode} mode.")

    for i in range(K):
        for j in range(i + 1, K):
            w = float(sim[i, j])
            if w >= thr:
                G.add_edge(str(domains[i]), str(domains[j]), weight=w)

    if G.number_of_edges() == 0:
        print("All similarities below threshold; no edges to draw.")
        return (None, None, None, G) if return_graph else (None, None, None)

    edge_w = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    ew_min, ew_max = edge_w.min(), edge_w.max()
    if ew_max > ew_min:
        layout_w = 0.1 + 0.9 * (edge_w - ew_min) / (ew_max - ew_min)
    else:
        layout_w = np.full_like(edge_w, 0.5)
    for idx, (u, v) in enumerate(G.edges()):
        G[u][v]["layout_weight"] = float(layout_w[idx])
    pos = nx.spring_layout(G, weight="layout_weight", seed=0, k=0.3, iterations=50)
    nodes_list = list(G.nodes())

    if color_mode == "dti":
        if dti_series is None:
            raise ValueError("color_mode='dti' but no DTI information found; check obs columns.")
        dti_vals = np.array([G.nodes[d]["DTI"] for d in nodes_list], dtype=float)
        if np.isnan(dti_vals).any():
            mask = ~np.isnan(dti_vals)
            fill = np.nanmean(dti_vals[mask]) if mask.any() else 0.5
            dti_vals[np.isnan(dti_vals)] = fill
        dti_min, dti_max = dti_vals.min(), dti_vals.max()
        node_color_vals_num = (dti_vals - dti_min) / (dti_max - dti_min) if dti_max > dti_min else np.zeros_like(dti_vals)
        cmap = cm.get_cmap("inferno")
        node_colors = node_color_vals_num
        rgba = cmap(node_color_vals_num)
        show_colorbar = True
    else:
        cl_vals = np.array([G.nodes[d]["CLUSTER"] for d in nodes_list], dtype=object)
        cl_vals_str = pd.Series(cl_vals, dtype="category").astype(str).values
        node_colors = [cluster_color_map.get(c, "#808080") for c in cl_vals_str]
        rgb = np.array([to_rgb(c) for c in node_colors])
        rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)
        show_colorbar = False
        cmap = None

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    ew_min, ew_max = edge_weights.min(), edge_weights.max()
    if ew_max > ew_min:
        edge_widths = max_edge_width * (edge_weights - ew_min) / (ew_max - ew_min)
    else:
        edge_widths = np.full_like(edge_weights, max_edge_width / 2)

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }

    with plt.rc_context(style):
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if color_mode == "dti":
            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=cmap,
                alpha=0.8,
                ax=ax,
            )
        else:
            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                ax=ax,
            )
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.8, ax=ax)

        luminance = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
        domain_to_idx = {d: i for i, d in enumerate(nodes_list)}
        for node, (x, y) in pos.items():
            idx = domain_to_idx[node]
            text_color = "white" if luminance[idx] < 0.5 else "black"
            ax.text(x, y, str(node), fontsize=7, ha="center", va="center", color=text_color)

        if show_colorbar:
            cbar = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("DTI", fontsize=10)
            cbar.outline.set_linewidth(0.5)

        if legend and color_mode == "cluster":
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", label=cat,
                       markerfacecolor=cluster_color_map.get(cat, "#808080"), markersize=8)
                for cat in cluster_cats
            ]
            ax.legend(handles=legend_elements, title="Coarse cluster", loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, title_fontsize=9)

        title = "Domain graph (DTI)" if color_mode == "dti" else f"Domain graph ({domain_key}_coarse)"
        ax.set_title(title, fontsize=12, pad=12, weight='medium')
        ax.axis("off")

        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
            elif show:
                plt.show()

    if return_graph:
        return fig, ax, saved_path, G
    return fig, ax, saved_path


def plot_domain_boundary_graph(
    adata,
    domain_key="fine",
    title=None,
    node_size=300,
    min_width=1.0,
    max_width=3.0,
    cmap="viridis",
    node_color="skyblue",
    node_alpha=0.9,
    edge_alpha=0.8,
    layout_k=0.2,
    iterations=1000,
    figsize=(8, 6),
    output_dir=None,
    filename=None,
    show=False,
    close=None,
    dpi=600,
    return_graph=False,
):
    """Draw a spatial-adjacency boundary-strength graph between domains."""
    key = f"{domain_key}_boundary_strength"
    if key not in adata.uns:
        raise KeyError(f"adata.uns missing '{key}'.")
    if filename is None:
        filename = f"{domain_key}_boundary_graph.png"
    if title is None:
        title = f"{domain_key} boundary graph"

    boundary_strength = adata.uns[key].copy()
    required = {"domain_a", "domain_b", "strength"}
    if not required.issubset(boundary_strength.columns):
        raise ValueError(f"{key} must contain columns: {sorted(required)}")

    G = nx.Graph()
    for _, row in boundary_strength.iterrows():
        G.add_edge(str(row["domain_a"]), str(row["domain_b"]), weight=float(row["strength"]))

    if G.number_of_edges() == 0:
        print("boundary_strength contains no edges to draw.")
        return (None, None, None, G) if return_graph else (None, None, None)

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    w_min, w_max = edge_weights.min(), edge_weights.max()
    if w_max > w_min:
        widths = min_width + (edge_weights - w_min) / (w_max - w_min) * (max_width - min_width)
    else:
        widths = np.full_like(edge_weights, (min_width + max_width) / 2)

    pos = nx.spring_layout(G, k=layout_k, iterations=iterations, weight="weight", seed=0)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=node_alpha, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color=edge_weights,
        edge_cmap=plt.get_cmap(cmap),
        alpha=edge_alpha,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="black", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()

    saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
    if return_graph:
        return fig, ax, saved_path, G
    return fig, ax, saved_path


def plot_domain_gain_log(
    adata: ad.AnnData,
    uns_key: str = "domain_gain_log",
    gain_key: str = "best_gain",
    title: str = None,
    ylabel: str = None,
    figsize: tuple = (8, 4),
    output_dir: str = None,
    filename: str = None,
    show: bool = False,
    close=None,
    dpi: int = 600,
):
    """Plot one column from a domain-merge gain log stored in ``adata.uns``."""
    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns missing key: '{uns_key}'")

    df = adata.uns[uns_key]
    if gain_key not in df.columns:
        raise ValueError(f"DataFrame missing gain column: '{gain_key}'")
    if title is None:
        title = f"{gain_key} per domain merge step"
    if ylabel is None:
        ylabel = gain_key
    if filename is None:
        filename = f"{_safe_filename(gain_key)}_log.png"

    fig, ax = plt.subplots(figsize=figsize)
    x_data = df["step"] if "step" in df.columns else df.index
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(x_data, df[gain_key], marker="o", linestyle="-", color="#1f77b4", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Merge step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
    return fig, ax, saved_path


def plot_domain_tree(
    adata,
    figsize=(14, 3.5),       # Keep the original wide layout
    fontsize=10,           #
    cmap_name='inferno',
    output_dir=None,
    filename="Domain_Hierarchy_Tree_DTI.png",  # Recommended vector output
    show=False,
    close=None,
    dpi=600,
    ax=None,               # Supports external axes for composites
):
    """
    Three-level domain hierarchy tree with DTI-coloured edges.
    Nature-style formatting applied without changing original thicknesses.
    """
    domain_map_key="domain_map"
    coarse_map_key="domain_coarse_map"
    fine_dti_col="fine_DTI"
    domain_dti_col="domain_DTI"
    fine_col="fine"
    domain_col="domain"

    # Base node size
    node_size_base=220

    gap_fine_within_domain=0
    gap_between_domains=0.25
    gap_between_coarse=0.5
    edge_alpha=1
    show_colorbar=True

    if domain_map_key not in adata.uns or coarse_map_key not in adata.uns:
        raise KeyError("adata.uns must contain domain_map and domain_coarse_map.")
    if fine_dti_col not in adata.obs:
        raise KeyError(f"adata.obs missing column: {fine_dti_col}")
    if domain_dti_col not in adata.obs:
        raise KeyError(f"adata.obs missing column: {domain_dti_col}")
    if fine_col not in adata.obs:
        raise KeyError(f"adata.obs missing column: {fine_col}")

    domain_map = adata.uns[domain_map_key]
    coarse_map = adata.uns[coarse_map_key]

    def _s(x): return str(x)

    coarse_ids = sorted([_s(c) for c in coarse_map.keys()], key=lambda x: int(x) if x.isdigit() else x)
    normal_ids = sorted({_s(n) for normals in coarse_map.values() for n in normals},
                        key=lambda x: int(x) if x.isdigit() else x)
    fine_ids = sorted({_s(f) for fines in domain_map.values() for f in fines},
                      key=lambda x: int(x) if x.isdigit() else x)

    region_series = adata.obs[fine_col].astype(str)

    if domain_col in adata.obs:
        domain_series = adata.obs[domain_col].astype(str)
    else:
        region_to_domain = {}
        for d, regs in domain_map.items():
            d = _s(d)
            for r in regs:
                r = _s(r)
                if r in region_to_domain and region_to_domain[r] != d:
                    raise ValueError(f"fine {r} belongs to both domain {region_to_domain[r]} and {d}")
                region_to_domain[r] = d
        domain_series = region_series.map(region_to_domain).astype(str)

    region_dti = adata.obs.groupby(region_series, observed=True)[fine_dti_col].mean().to_dict()
    domain_dti = adata.obs.groupby(domain_series, observed=True)[domain_dti_col].mean().to_dict()

    nodes = {}
    for c in coarse_ids: nodes[f"C|{c}"] = {"label": c, "level": 0}
    for n in normal_ids: nodes[f"D|{n}"] = {"label": n, "level": 1}
    for f in fine_ids:   nodes[f"F|{f}"] = {"label": f, "level": 2}

    edges = []
    edge_value = {}
    for c, normals in coarse_map.items():
        u = f"C|{_s(c)}"
        for n in normals:
            v = f"D|{_s(n)}"
            edges.append((u, v))
            edge_value[(u, v)] = domain_dti.get(_s(n), np.nan)

    for n, fines in domain_map.items():
        u = f"D|{_s(n)}"
        for f in fines:
            v = f"F|{_s(f)}"
            edges.append((u, v))
            edge_value[(u, v)] = region_dti.get(_s(f), np.nan)

    tree_struct = {c: {} for c in coarse_ids}
    for c in coarse_ids:
        normals = [_s(n) for n in coarse_map.get(c, []) if _s(n) in normal_ids]
        for n in sorted(normals, key=lambda x: int(x) if x.isdigit() else x):
            fines = [_s(f) for f in domain_map.get(n, []) if _s(f) in fine_ids]
            tree_struct[c][n] = sorted(fines, key=lambda x: int(x) if x.isdigit() else x)

    pos, fine_x_map, normal_placeholder_x, current_x = {}, {}, {}, 0.0
    for c in coarse_ids:
        normals = list(tree_struct[c].keys())
        for n in normals:
            fines = tree_struct[c][n]
            if not fines:
                normal_placeholder_x[n] = current_x
                current_x += 1.0
            else:
                for f in fines:
                    fine_x_map[f"F|{f}"] = current_x
                    current_x += 1.0 + gap_fine_within_domain
            current_x += gap_between_domains
        current_x += gap_between_coarse

    for nid, info in nodes.items():
        if info["level"] == 2:
            pos[nid] = (fine_x_map.get(nid, np.nan), -2)

    for n in normal_ids:
        n_id = f"D|{n}"
        children = [f"F|{_s(f)}" for f in domain_map.get(n, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[n_id] = (float(np.mean(child_xs)), -1) if child_xs else (float(normal_placeholder_x.get(n, 0.0)), -1)

    for c in coarse_ids:
        c_id = f"C|{c}"
        children = [f"D|{_s(n)}" for n in coarse_map.get(c, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[c_id] = (float(np.mean(child_xs)), 0) if child_xs else (0.0, 0)

    xs = np.array([p[0] for p in pos.values() if not np.isnan(p[0])], dtype=float)
    if xs.size > 0 and xs.max() > xs.min():
        xmin, xmax = xs.min(), xs.max()
        for k in list(pos.keys()):
            if not np.isnan(pos[k][0]):
                pos[k] = ((pos[k][0] - xmin) / (xmax - xmin), pos[k][1])
    else:
        for k in list(pos.keys()):
            if not np.isnan(pos[k][0]):
                pos[k] = (0.5, pos[k][1])

    vals = np.array([v for v in edge_value.values() if not np.isnan(v)], dtype=float)
    norm = Normalize(vmin=0, vmax=1) if vals.size == 0 else Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap(cmap_name)

    # ==========================================
    # Apply typography without changing figsize.
    # ==========================================
    style = {
        "figure.figsize": figsize,
        "figure.dpi": 600,             # Control notebook preview size
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    }

    with plt.rc_context(style):
        is_standalone = (ax is None)
        if is_standalone:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(100)
        else:
            fig = ax.get_figure()

        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if np.isnan(x1) or np.isnan(x2):
                continue
            val = edge_value.get((u, v), np.nan)

            color = "#CFCFCF" if np.isnan(val) else cmap(norm(val))
            t = None if np.isnan(val) else float(norm(val))

            is_coarse_to_domain = u.startswith("C|") and v.startswith("D|")
            is_domain_to_fine   = u.startswith("D|") and v.startswith("F|")

            # Ribbon width logic is unchanged.
            if np.isnan(val):
                wpx = 12 if is_coarse_to_domain else 7
            else:
                if is_coarse_to_domain:
                    wpx = 5 + 12 * t
                elif is_domain_to_fine:
                    wpx = 2 + 4 * t
                else:
                    wpx = 6 + 8 * t

            _draw_sankey_ribbon(
                ax, (x1, y1), (x2, y2),
                color=color, alpha=edge_alpha, width_px=wpx, n=25, zorder=1, outline=True
            )

        colors = {0: "#2E8B57", 1: "#E47734", 2: "#6A5ACD"}
        for nid, (x, y) in pos.items():
            level = nodes[nid]["level"]
            label = nodes[nid]["label"]
            color = colors.get(level, "black")

            # Node-size logic is unchanged.
            if level == 0:
                size = node_size_base * 1.6
            elif level == 1: size = node_size_base * 1.25
            else: size = node_size_base * 1.0

            ax.scatter(x, y, s=size, c=color, edgecolor="black", linewidth=0.6, zorder=2)

            current_fs = fontsize if len(label) <= 2 else max(6, fontsize - 2)
            ax.text(x, y, label, ha="center", va="center", color="white", fontsize=current_fs, zorder=3)

        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(["Coarse", "Domain", "Fine"], fontsize=12)

        for spine in ax.spines.values(): spine.set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.tick_params(left=False)

        if show_colorbar and vals.size > 0:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("DTI", fontsize=11)
            cbar.outline.set_linewidth(0.5)

        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if '_save_show_close' in globals():
                saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
            else:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    outpath = os.path.join(output_dir, filename)
                    plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
                    saved_path = outpath
                if show:
                    plt.show()

    return fig, ax, saved_path


def _cast_group_to_dtype(adata, key, group_str):
    """
    Cast a group label string to the dtype of ``obs[key]`` categories.
    """
    s = str(group_str)
    ser = adata.obs[key]

    # Category values use the categories dtype.
    if str(ser.dtype) == "category":
        cats = ser.cat.categories
        if np.issubdtype(cats.dtype, np.integer):
            return int(s)
        return s

    # Non-category values preserve the original dtype when possible.
    dt = ser.dtype
    if np.issubdtype(dt, np.integer):
        return int(s)
    return s


def plot_domain_tree_spatial(
    adata,
    spot_size=100,
    width_per_node=1.2,
    height=10,
    figsize=None,
    inset_size=3,
    gene_tree=None,
    cmap_name="inferno",
    output_dir=None,
    filename=None,
    palette=None,
    show=False,
    close=None,
    dpi=600,
    img_key=None,
    ax=None,             # Supports external axes
):
    """
    Three-level domain hierarchy tree with spatial plot insets.
    Includes Nature-style DPI locks and editable PDF font support.
    """
    _adata=adata.copy()

    domain_map_key="domain_map"
    coarse_map_key="domain_coarse_map"
    fine_dti_col="fine_DTI"
    domain_dti_col="domain_DTI"
    fine_col="fine"

    fontsize=11

    gap_fine_within_domain=1.5
    gap_between_domains=0.8
    gap_between_coarse=1.5
    y_step=5.0

    edge_alpha=0.9
    show_colorbar=True

    if domain_map_key not in _adata.uns or coarse_map_key not in _adata.uns:
        raise KeyError(f"mapping tables not found in adata.uns: {domain_map_key} or {coarse_map_key}")
    if fine_col not in _adata.obs:
        raise KeyError(f"adata.obs missing column: {fine_col}")
    if fine_dti_col not in _adata.obs:
        raise KeyError(f"adata.obs missing column: {fine_dti_col}")
    if domain_dti_col not in _adata.obs:
        raise KeyError(f"adata.obs missing column: {domain_dti_col}")

    domain_map_raw = _adata.uns[domain_map_key]
    coarse_map_raw = _adata.uns[coarse_map_key]

    def _s(x): return str(x)
    def _sort_key(x):
        x = str(x)
        return int(x) if x.isdigit() else x

    domain_map = { _s(d): [_s(f) for f in fs] for d, fs in domain_map_raw.items() }
    coarse_map = { _s(c): [_s(d) for d in ds] for c, ds in coarse_map_raw.items() }

    fine_to_domain = {}
    for d, fs in domain_map.items():
        for f in fs:
            if f in fine_to_domain and fine_to_domain[f] != d:
                raise ValueError(f"fine {f} belongs to both domain {fine_to_domain[f]} and {d}")
            fine_to_domain[f] = d

    dom_to_coarse = {}
    for c, ds in coarse_map.items():
        for d in ds:
            if d in dom_to_coarse and dom_to_coarse[d] != c:
                raise ValueError(f"domain {d} belongs to both coarse {dom_to_coarse[d]} and {c}")
            dom_to_coarse[d] = c

    d_col, c_col = "tmp_tree_domain", "tmp_tree_coarse"
    region_series = _adata.obs[fine_col].astype(str)
    _adata.obs[d_col] = region_series.map(fine_to_domain).astype(str)
    _adata.obs[c_col] = _adata.obs[d_col].map(dom_to_coarse).astype(str)

    region_dti = _adata.obs.groupby(region_series, observed=True)[fine_dti_col].mean().to_dict()
    domain_dti = _adata.obs.groupby(_adata.obs[d_col].astype(str), observed=True)[domain_dti_col].mean().to_dict()

    coarse_ids = sorted(list(coarse_map.keys()), key=_sort_key)
    normal_ids = sorted({d for ds in coarse_map.values() for d in ds}, key=_sort_key)
    fine_ids   = sorted({f for fs in domain_map.values() for f in fs}, key=_sort_key)

    tree_struct = {c: {} for c in coarse_ids}
    for c in coarse_ids:
        normals = [n for n in coarse_map.get(c, []) if n in normal_ids]
        for n in sorted(normals, key=_sort_key):
            fines = [f for f in domain_map.get(n, []) if f in fine_ids]
            tree_struct[c][n] = sorted(fines, key=_sort_key)

    pos, fine_x_map, normal_placeholder_x, current_x = {}, {}, {}, 0.0
    for c in coarse_ids:
        normals = list(tree_struct[c].keys())
        for n in normals:
            fines = tree_struct[c][n]
            if not fines:
                normal_placeholder_x[n] = current_x
                current_x += 1.0
            else:
                for f in fines:
                    fine_x_map[f"F|{f}"] = current_x
                    current_x += 1.0 + gap_fine_within_domain
            current_x += gap_between_domains
        current_x += gap_between_coarse

    for f in fine_ids: pos[f"F|{f}"] = (fine_x_map.get(f"F|{f}", np.nan), -2 * y_step)

    for n in normal_ids:
        n_id = f"D|{n}"
        children = [f"F|{f}" for f in domain_map.get(n, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[n_id] = (float(np.mean(child_xs)), -1 * y_step) if child_xs else (float(normal_placeholder_x.get(n, 0.0)), -1 * y_step)

    for c in coarse_ids:
        c_id = f"C|{c}"
        children = [f"D|{n}" for n in coarse_map.get(c, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[c_id] = (float(np.mean(child_xs)), 0.0) if child_xs else (0.0, 0.0)

    xs = np.array([p[0] for p in pos.values() if not np.isnan(p[0])], dtype=float)
    if xs.size > 0 and xs.max() > xs.min():
        xmin, xmax = xs.min(), xs.max()
        for k in list(pos.keys()):
            if not np.isnan(pos[k][0]):
                pos[k] = ((pos[k][0] - xmin) / (xmax - xmin), pos[k][1])
    else:
        for k in list(pos.keys()):
            if not np.isnan(pos[k][0]):
                pos[k] = (0.5, pos[k][1])

    edges, edge_value = [], {}
    for c, normals in coarse_map.items():
        u = f"C|{c}"
        for n in normals:
            v = f"D|{n}"
            edges.append((u, v))
            edge_value[(u, v)] = domain_dti.get(n, np.nan)

    for n, fines in domain_map.items():
        u = f"D|{n}"
        for f in fines:
            v = f"F|{f}"
            edges.append((u, v))
            edge_value[(u, v)] = region_dti.get(f, np.nan)

    vals = np.array([v for v in edge_value.values() if not np.isnan(v)], dtype=float)
    norm = Normalize(vmin=0, vmax=1) if vals.size == 0 else Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap(cmap_name)

    total_x_units = 0.0
    for c in coarse_ids:
        for n in sorted(coarse_map.get(c, []), key=_sort_key):
            fines = sorted(domain_map.get(n, []), key=_sort_key)
            if not fines:
                total_x_units += 1.0
            else: total_x_units += len(fines)
            total_x_units += gap_between_domains
        total_x_units += gap_between_coarse
    dynamic_width = max(10, total_x_units * width_per_node)

    # ==========================================
    # Inject style, limit preview size, and use external axes.
    # ==========================================
    style = {
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,            # Keep PDF text editable
        "ps.fonttype": 42
    }

    with plt.rc_context(style):
        is_standalone = (ax is None)
        if is_standalone:
            if figsize is None:
                figsize = (dynamic_width, height)
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(100)
        else:
            fig = ax.get_figure()

        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if np.isnan(x1) or np.isnan(x2):
                continue

            val = edge_value.get((u, v), np.nan)
            if np.isnan(val):
                color = "#CFCFCF"
                t = None
            else:
                color = cmap(norm(val))
                t = float(norm(val))

            is_coarse_to_domain = u.startswith("C|") and v.startswith("D|")
            is_domain_to_fine   = u.startswith("D|") and v.startswith("F|")

            if np.isnan(val):
                wpx = 12 if is_coarse_to_domain else 7
            else:
                if is_coarse_to_domain:
                    wpx = 10 + 12 * t
                elif is_domain_to_fine: wpx = 4 + 4 * t
                else: wpx = 6 + 8 * t

            _draw_sankey_ribbon(
                ax, (x1, y1), (x2, y2),
                color=color, alpha=edge_alpha, width_px=wpx, n=25, zorder=1, outline=True
            )

        colors = {0: "#2E8B57", 1: "#E47734", 2: "#6A5ACD"}

        for nid, (nx, ny) in pos.items():
            if nid.startswith("C|"):
                lvl, col = 0, "domain_coarse"
                label = nid.split("|", 1)[1]
                edge_id = f"coarse:{label}"
            elif nid.startswith("D|"):
                lvl, col = 1, "domain"
                label = nid.split("|", 1)[1]
                edge_id = f"domain:{label}"
            else:
                lvl, col = 2, fine_col
                label = nid.split("|", 1)[1]
                edge_id = f"fine:{label}"

            sub_ax = inset_axes(
                ax, width="100%", height="100%",
                bbox_to_anchor=(nx - inset_size / 2, ny - inset_size / 2, inset_size, inset_size),
                bbox_transform=ax.transData, borderpad=0
            )

            if gene_tree is not None:
                # Assume calculate_edge_scores is defined externally.
                score_values = TGL_module.calculate_edge_scores(_adata, gene_tree, edge_id, Knee_S=3)
                score_name = f"tmp_score_{edge_id.replace(':', '_')}"
                _adata.obs[score_name] = score_values
                sc.pl.spatial(_adata, color=score_name, img_key=img_key, spot_size=spot_size,
                              frameon=False, ax=sub_ax, show=False, title='', cmap="RdYlBu_r",
                              legend_loc=None, colorbar_loc=None)
            else:
                gid = _cast_group_to_dtype(_adata, col, label)  # External helper
                sc.pl.spatial(_adata, color=col, img_key=img_key, groups=[gid], spot_size=spot_size,
                              frameon=False, ax=sub_ax, show=False, title='', legend_loc=None, colorbar_loc=None)

            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_xlabel("")
            sub_ax.set_ylabel("")
            sub_ax.set_title("")

            for spine in sub_ax.spines.values():
                spine.set_edgecolor(colors.get(lvl, "black"))
                spine.set_linewidth(2)

            s_label = str(label)
            current_fs = fontsize if len(s_label) <= 2 else max(6, fontsize - 2)

            ax.text(
                nx, ny+1.8, s_label, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="circle,pad=0.28", fc=colors.get(lvl, "black"), ec="black", lw=0.6, alpha=0.95)
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-2 * y_step - inset_size, 0 + inset_size)

        ax.set_yticks([0, -1 * y_step, -2 * y_step])
        ax.set_yticklabels(["Coarse", "Domain", "Fine"], fontsize=14, fontweight="bold")

        for s in ax.spines.values(): s.set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.tick_params(left=False)

        if show_colorbar and vals.size > 0:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
            cbar.set_label("DTI Intensity", fontsize=12)
            cbar.outline.set_linewidth(0.5)

        saved_path = None
        if is_standalone:
            fig.tight_layout()
            if filename is None:
                filename = "gene_tree_spatial.pdf" if gene_tree is not None else "domain_tree_spatial.pdf"

            if '_save_show_close' in globals():
                saved_path = _save_show_close(fig, output_dir, filename, dpi=dpi, show=show, close=close)
            else:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    outpath = os.path.join(output_dir, filename)
                    plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
                    saved_path = outpath
                if show:
                    plt.show()

    return fig, ax, saved_path
