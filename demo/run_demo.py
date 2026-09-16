
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = DEMO_DIR / "data" / "E9.5_E1S1.h5ad"
DEFAULT_OUTPUT = DEMO_DIR / "output"
SPOT_SIZE = 1.3
FIGURE_DPI = 600
TARGET_N_DOMAIN = 27
EXPECTED_IMAGES = (
    "boundary_and_domain_DTI.png",
    "multiscale_domains.png",
    "domain_graph.png",
    "Domain_Hierarchy_Tree_DTI.png",
    "spatial_module_score_domain_11.png",
    "gradient_bar_domain_11.png",
)
PALETTE = [
    "#7E1035", "#EF6C4A", "#FBDE8E", "#E5F29B", "#6AC0A4", "#5E519F",
    "#FF7F50", "#FFD700", "#8A2BE2", "#00CED1", "#ADFF2F", "#FF69B4",
    "#FF8C00", "#40E0D0", "#DA70D6", "#9ACD32", "#FF1493", "#FFA500",
    "#20B2AA", "#BA55D3", "#7FFF00", "#FF69B4", "#FF6347", "#48D1CC",
    "#C71585", "#32CD32", "#FF1493", "#FF4500", "#00FA9A", "#8B008B",
    "#1A5276", "#1E8449", "#922B21", "#76448A", "#D4AC0D", "#117864",
    "#A04000", "#2E4053", "#00FF00", "#FF00FF", "#00FFFF", "#FFFF00",
    "#FF3366", "#33FFCC", "#CCFF33", "#FF9933", "#C0C0C0", "#5499C7",
    "#48C9B0", "#F4D03F", "#EB984E", "#AF7AC5", "#52BE80", "#EC7063",
    "#A9DFBF", "#AED6F1", "#F9E79F", "#F5B7B1", "#D2B4DE", "#A2D9CE",
    "#FAD7A0", "#ABB2B9", "#000080", "#800000", "#008080", "#800080",
    "#BC8F8F", "#4682B4", "#D2691E", "#9FE2BF",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four-stage mouse embryo demo and save six figures."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help=f"Input .h5ad file (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Figure directory (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def save_figure(fig, path: Path, plt) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI, pad_inches=0)
    plt.close(fig)


def draw_runall_figures(adata, gene_tree, output_dir: Path, sc, plot, tgl, plt) -> None:
    # 1. Boundary probability and domain-level DTI.
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    plot.plot_spatial(
        adata, colors=["p_edge"], titles=["p_edge (boundary probability)"],
        spot_size=SPOT_SIZE, cmap="inferno", colorbar_loc="right",
        ax=left, show=False,
    )
    plot.plot_spatial(
        adata, colors=["domain_DTI"], titles=["Domain DTI"],
        spot_size=SPOT_SIZE, cmap="inferno", colorbar_loc="right",
        ax=right, show=False,
    )
    save_figure(fig, output_dir / EXPECTED_IMAGES[0], plt)

    # 2. Fine, domain and coarse spatial hierarchy.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, key, title in zip(
        axes, ("fine", "domain", "domain_coarse"),
        ("Level: fine", "Level: domain", "Level: coarse"),
    ):
        plot.plot_spatial(
            adata, colors=[key], titles=[title], spot_size=SPOT_SIZE,
            palette=PALETTE, colorbar_loc=None, legend_loc="right margin",
            ax=ax, show=False,
        )
    save_figure(fig, output_dir / EXPECTED_IMAGES[1], plt)

    # 3. Domain graph in DTI and cluster modes.
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    plot.plot_domain_graph(
        adata, domain_key="domain", min_sim_quantile=0.95,
        color_mode="dti", ax=left, show=False,
    )
    plot.plot_domain_graph(
        adata, domain_key="domain", min_sim_quantile=0.95,
        color_mode="cluster", legend=True, ax=right, show=False,
    )
    save_figure(fig, output_dir / EXPECTED_IMAGES[2], plt)

    # 4. Three-level domain hierarchy tree.
    plot.plot_domain_tree(
        adata, figsize=(14, 3.5), show=False, close=True,
        output_dir=str(output_dir), filename=EXPECTED_IMAGES[3],
    )

    # 5. Spatial module score for the same domain:11 branch as RunAll.
    target_edge_id = "domain:11"
    if target_edge_id not in gene_tree["edge_rank_bc"]:
        raise RuntimeError(f"RunAll target branch {target_edge_id} was not produced.")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    sc.pl.spatial(
        adata, color="domain", groups=["11"], title="domain 11",
        spot_size=SPOT_SIZE, ax=left, frameon=False, show=False,
    )
    tgl.plot_gene_weights_spatial(
        adata, gene_tree, edge_id=target_edge_id, res_key="edge_rank_bc",
        spot_size=SPOT_SIZE, Knee_S=1, show_knee_plot=False,
        cmap="RdYlBu_r", ax=right, show=False,
    )
    right.set_title("Predicted Spatial Module Score", fontsize=12)
    save_figure(fig, output_dir / EXPECTED_IMAGES[4], plt)

    # 6. Top gene weights for that branch.
    tgl.plot_edge_top_genes_gradient(
        gene_tree, edge_id=target_edge_id, top_n=20, key="edge_rank_bc",
        cmap_name="RdBu_r", figure=(7, 5), show=False,
        out_dir=str(output_dir),
    )
    plt.close("all")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = args.data.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not data_path.is_file():
        print(f"Demo dataset not found: {data_path}", file=sys.stderr)
        return 2
    if data_path.suffix.lower() != ".h5ad":
        print(f"Expected an .h5ad input: {data_path}", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc
    import stEDGE
    from stEDGE import TGL_module, plot

    started = time.perf_counter()
    print(f"stEDGE version: {stEDGE.__version__}")
    adata = sc.read_h5ad(data_path)
    if "Region" not in adata.obs:
        raise ValueError("Input adata.obs must contain 'Region'.")
    if "spatial" not in adata.obsm or adata.obsm["spatial"].shape != (adata.n_obs, 2):
        raise ValueError("Input adata.obsm['spatial'] must have two coordinates per spot.")
    if not np.isfinite(adata.obsm["spatial"]).all():
        raise ValueError("Input spatial coordinates contain non-finite values.")

    # The RunAll preprocessing; raw counts need no copy because no AnnData is exported.
    adata = adata[adata.obs["Region"].notna()].copy()
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_genes(adata, min_counts=5)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable", svd_solver="arpack")
    print(f"Preprocessing complete: {adata.n_obs} spots x {adata.n_vars} genes")

    output_dir.mkdir(parents=True, exist_ok=True)
    runner = stEDGE.StEDGE(output_dir=str(output_dir), verbose=True)
    result = runner.run_full_pipeline(
        adata, stages=(1, 2, 3),
        stage1_kwargs=dict(
            resolution_range=(1.0, 3.0, 0.1), n_neighbors=13,
            leiden_flavor="leidenalg", clustering_backend="threading",
            consensus_backend="threading", boundary_n_neighbors=7,
        ),
        stage2_kwargs=dict(
            seed_quantile=0.45, seed_n_neighbors=6, min_seed_size=5,
            global_thr_k=0.45, con_th=0.60, detect_unassigned=True,
            con_th_inner=0.70, con_min_size=20,
        ),
        stage3_kwargs=dict(
            target_n_domains=TARGET_N_DOMAIN, min_sim_quantile=0.95,
            coarse_resolution=1.0,
        ),
    )
    adata = result["adata"]
    print(f"Fine: {adata.obs['fine'].nunique()}  |  "
          f"Domain: {adata.obs['domain'].nunique()}  |  "
          f"Coarse: {adata.obs['domain_coarse'].nunique()}")

    # The wrapper's Stage 4 always exports rankings. Run the same public
    # calculation without out_dir so this demo only saves RunAll's figures.
    gene_tree = TGL_module.rank_multiscale_genes_tree(
        adata, max_genes=3000, lam=0.1, lam_excl=0.1,
        max_iter=200, top_n=50, out_dir=None,
    )
    draw_runall_figures(adata, gene_tree, output_dir, sc, plot, TGL_module, plt)
    print(f"Demo complete. Six figures saved to: {output_dir}")
    print(f"Elapsed time: {time.perf_counter() - started:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
