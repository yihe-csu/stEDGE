"""stEDGE — Spatial Transcriptomics Edge-guided Domain Discovery.

Four-stage architecture (see Methods):

    Stage 1 — Consensus-based boundary modeling
        Multi-resolution consensus clustering → boundary probability field p_edge.

    Stage 2 — Edge-guided fine-domain discovery
        Seed detection → boundary-preserving region growing → consensus propagation
        → label propagation → fine-domain assignment.

    Stage 3 — Transition-aware hierarchical refinement
        DTI scoring → domain similarity → boundary strength → DTI-guided merging
        → domain / coarse-domain hierarchy.

    Stage 4 — Tree-guided multiscale gene attribution
        Node-split task construction → tree-group lasso + exclusive lasso optimisation
        → multiscale gene rankings.

Typical usage::

    import stEDGE

    stedge = stEDGE.StEDGE(output_dir="./results", verbose=True)
    adata = stedge.run_full_pipeline(adata)

Or call stages individually::

    adata = stedge.run_stage1(adata, resolution_range=(0.1, 2, 0.1))
    adata = stedge.run_stage2(adata, seed_quantile=0.15)
    adata = stedge.run_stage3(adata, min_gain=1.0)
    gene_tree = stedge.run_stage4(adata, lam=0.2)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import scanpy as sc

from . import plot
from . import preprocess
from . import TGL_module
from . import utils


# ============================================================================
# Default parameters (aligned with Methods section)
# ============================================================================

DEFAULTS_STAGE1 = dict(
    # ---- consensus clustering ----
    method="leiden",
    resolution_range=(0.1, 2.0, 0.1),
    n_neighbors=15,
    use_rep="X_pca",
    n_jobs=-1,
    leiden_flavor="igraph",
    clustering_backend  = "loky",
    consensus_backend = "loky",
    # ---- boundary probability ----
    boundary_n_neighbors=7,
)

DEFAULTS_STAGE2 = dict(
    # ---- seed detection (Eq. 3) ----
    seed_quantile=0.10,
    seed_n_neighbors=6,
    min_seed_size=5,
    # ---- region growing (Eq. 4-5) ----
    global_thr_k=0.25,
    use_global_cap=True,
    hole_fill=True,
    hole_fill_thr="global",
    grad_k=12,
    grad_norm=False,
    # ---- consensus propagation (Eq. 6) ----
    con_th=0.7,
    detect_unassigned=True,
    con_th_inner=0.7,
    con_min_size=20,
    # ---- label propagation ----
    lp_n_components=50,
    lp_n_neighbors=7,
    spatial_weight=0.3,
    # ---- ablation (internal / experimental) ----
    ablation=None,
)

DEFAULTS_STAGE3 = dict(
    # ---- DTI & similarity ----
    domain_key="fine",
    # ---- DTI-guided merging (Eq. 10) ----
    new_domain_key="domain",
    min_gain=1.0,
    max_dti_threshold=0.65,
    w_sim=1.0,
    w_dti=0.5,
    w_boundary=0.5,
    target_n_domains=None,
    # ---- boundary strength ----
    boundary_n_neighbors=6,
    # ---- coarse-domain discovery ----
    min_sim_quantile=0.5,
    coarse_resolution=1.0,
)

DEFAULTS_STAGE4 = dict(
    max_genes=3000,
    lam=0.2,
    lam_excl=0.1,
    max_iter=200,
    tol=1e-4,
    step_size=None,
    top_n=50,
    random_state=0,
    use_tree=True,
)


# ============================================================================
# StEDGE class
# ============================================================================


class StEDGE:
    """Spatial transcriptomics edge-guided domain discovery pipeline.

    Parameters
    ----------
    output_dir : str
        Root directory for intermediate and final outputs.
    verbose : bool
        Whether to print progress messages.
    """

    # Defaults (overridable per-call via kwargs)
    defaults_stage1 = DEFAULTS_STAGE1.copy()
    defaults_stage2 = DEFAULTS_STAGE2.copy()
    defaults_stage3 = DEFAULTS_STAGE3.copy()
    defaults_stage4 = DEFAULTS_STAGE4.copy()

    def __init__(self, output_dir: str = "./stEDGE_output", verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    def _merge(self, defaults: dict, kwargs: dict) -> dict:
        """Return a merged parameter dict (kwargs override defaults)."""
        merged = defaults.copy()
        merged.update({k: v for k, v in kwargs.items() if k in defaults})
        return merged

    # ------------------------------------------------------------------
    # Stage 1 — Consensus-based boundary modeling
    # ------------------------------------------------------------------

    def run_stage1(self, adata: sc.AnnData, **kwargs) -> sc.AnnData:
        """Consensus-based boundary modeling.

        Computes a multi-resolution consensus matrix and derives a continuous
        boundary probability field p_edge (Eqs. 1-2 in Methods).

        Parameters
        ----------
        adata : AnnData
            Must contain ``.X``, ``obsm["X_pca"]``, and ``obsm["spatial"]``.
        method : str
            Clustering method: ``"leiden"`` or ``"kmeans"``.
        resolution_range : tuple
            (start, stop, step) for the resolution grid.
        n_neighbors : int
            Number of neighbours for the expression graph.
        use_rep : str
            Key in ``obsm`` for the input representation.
        n_jobs : int
            Number of parallel workers (-1 = all cores).
        boundary_n_neighbors : int
            Number of spatial neighbours for the boundary likelihood window.

        Returns
        -------
        adata : AnnData
            Modified in-place with ``obsm["consensus_freq"]`` and
            ``obs["p_edge"]``.
        """
        p = self._merge(self.defaults_stage1, kwargs)

        if self.verbose:
            print("=" * 60)
            print("  Stage 1 — Consensus-based boundary modeling")
            print("=" * 60)
            print("  Computing multi-resolution consensus matrix ...")

        preprocess.consensus_clustering(
            adata,
            method=p["method"],
            resolution_range=p["resolution_range"],
            n_neighbors=p["n_neighbors"],
            use_rep=p["use_rep"],
            clustering_backend=p["clustering_backend"],
            consensus_backend=p["consensus_backend"],
            n_jobs=p["n_jobs"],
            leiden_flavor=p["leiden_flavor"],
        )

        if self.verbose:
            print("  Estimating boundary probability field ...")

        utils.compute_boundary_probability(
            adata,
            n_neighbors=p["boundary_n_neighbors"],
        )

        if self.verbose:
            print(f"  Stage 1 complete — p_edge stored in adata.obs['p_edge'].\n")

        return adata

    # ------------------------------------------------------------------
    # Stage 2 — Edge-guided fine-domain discovery
    # ------------------------------------------------------------------

    def run_stage2(self, adata: sc.AnnData, **kwargs) -> sc.AnnData:
        """Edge-guided fine-domain discovery.

        Detects stable seed regions in low-boundary areas, performs competitive
        boundary-preserving region growing, propagates domains via consensus
        support, and completes fine-domain labels via spatially constrained
        label propagation (Eqs. 3-6 in Methods).

        Parameters
        ----------
        adata : AnnData
            Must contain ``obsm["consensus_freq"]``, ``obs["p_edge"]``, and
            ``obsm["spatial"]``.
        seed_quantile : float
            Fraction of lowest-p_edge spots used as seed candidates.
        seed_n_neighbors : int
            Spatial neighbours for the seed connectivity graph.
        min_seed_size : int
            Minimum seed size to retain.
        global_thr_k : float
            IQR multiplier for the global boundary threshold (Eq. 5).
        use_global_cap : bool
            Cap per-seed thresholds at the global threshold.
        hole_fill : bool
            Fill isolated unlabeled holes after region growing.
        hole_fill_thr : str
            ``"global"`` or ``"seed"``.
        grad_k : int
            Number of spatial neighbours for gradient estimation (Eq. 4).
        grad_norm : bool
            Whether to standardise coordinates before gradient estimation.
        con_th : float
            Consensus-support threshold for domain propagation (Eq. 6).
        detect_unassigned : bool
            Whether to detect new domains from remaining unassigned spots.
        con_th_inner : float
            Consensus threshold for new-domain detection.
        con_min_size : int
            Minimum size for new domains.
        lp_n_components : int
            PCA components for label propagation.
        lp_n_neighbors : int
            kNN neighbours for label propagation.
        spatial_weight : float
            Weight of spatial coordinates relative to expression PCs.
        ablation : str or None
            Experimental: skip or replace a sub-step for ablation studies.

        Returns
        -------
        adata : AnnData
            Modified in-place with ``obs["fine"]`` and intermediate columns.
        """
        p = self._merge(self.defaults_stage2, kwargs)
        ablation = p.pop("ablation")

        if self.verbose:
            print("=" * 60)
            print("  Stage 2 — Edge-guided fine-domain discovery")
            print("=" * 60)

        # ---- boundary → seed → region growing ----
        if ablation == "no_boundary":
            self._log("  [ABLATION] Skipping boundary/seed/region growing; using pre_domain.")
            if "pre_domain" in adata.obs:
                adata.obs["region_label"] = (
                    adata.obs["pre_domain"].astype(int).astype("category")
                )
            else:
                cols = adata.obsm["clusters_results"].columns
                mid_col = cols[len(cols) // 2]
                adata.obs["region_label"] = pd.Categorical(
                    adata.obsm["clusters_results"][mid_col].values.astype(int)
                )
                self._log(f"  [FALLBACK] pre_domain missing; using {mid_col} as region_label.")
        else:
            if ablation != "no_region_growing":
                self._log("  Detecting seed regions ...")
                utils.find_seed_regions(
                    adata,
                    seed_quantile=p["seed_quantile"],
                    n_neighbors=p["seed_n_neighbors"],
                    min_seed_size=p["min_seed_size"],
                )
                self._log("  Performing boundary-preserving region growing ...")
                utils.compute_gradient(
                    adata,
                    k=p["grad_k"],
                    norm=p["grad_norm"],
                )
                utils.region_growing_segmentation(
                    adata,
                    output_dir=self.output_dir,
                    global_thr_k=p["global_thr_k"],
                    use_global_cap=p["use_global_cap"],
                    hole_fill=p["hole_fill"],
                    hole_fill_thr=p["hole_fill_thr"],
                )
            else:
                self._log("  Detecting seed regions ...")
                utils.find_seed_regions(
                    adata,
                    seed_quantile=p["seed_quantile"],
                    n_neighbors=p["seed_n_neighbors"],
                    min_seed_size=p["min_seed_size"],
                )
                self._log("  [ABLATION] Skipping region growing; using seeds as region_label.")
                adata.obs["region_label"] = (
                    adata.obs["labeled_seeds"].astype(int).astype("category")
                )

        # ---- consensus-based domain propagation (Eq. 6) ----
        self._log("  Propagating domains via consensus support ...")
        utils.propagate_domains_from_regions(
            adata,
            con_th=p["con_th"],
            detect_unassigned=p["detect_unassigned"],
            con_th_inner=p["con_th_inner"],
            min_size=p["con_min_size"],
        )

        # ---- label propagation → fine ----
        if ablation == "no_label_prop":
            self._log("  [ABLATION] Skipping label propagation; using con_region as fine.")
            adata.obs["fine"] = adata.obs["con_region"].astype(int).astype("category")
        elif ablation == "label_spreading":
            self._log("  Completing fine labels via label spreading ...")
            utils.label_spreading_with_pca(
                adata,
                n_components=p["lp_n_components"],
                n_neighbors=p["lp_n_neighbors"],
                spatial_weight=p["spatial_weight"],
            )
        else:
            self._log("  Completing fine labels via label propagation ...")
            utils.label_propagation_with_pca(
                adata,
                n_components=p["lp_n_components"],
                n_neighbors=p["lp_n_neighbors"],
                spatial_weight=p["spatial_weight"],
            )

        if self.verbose:
            n_fine = adata.obs["fine"].nunique()
            print(f"  Stage 2 complete — {n_fine} fine domains.\n")

        return adata

    # ------------------------------------------------------------------
    # Stage 3 — Transition-aware hierarchical refinement
    # ------------------------------------------------------------------

    def run_stage3(self, adata: sc.AnnData, **kwargs) -> sc.AnnData:
        """Transition-aware hierarchical refinement of spatial domains.

        Computes domain transition indices (DTI), inter-domain similarity and
        boundary strength, then performs DTI-guided merging and coarse-domain
        discovery (Eqs. 7-10 in Methods).

        Parameters
        ----------
        adata : AnnData
            Must contain ``obs["fine"]``, ``obsm["consensus_freq"]``.
        domain_key : str
            Key in ``obs`` for the input fine-domain labels.
        new_domain_key : str
            Key for the merged domain labels.
        min_gain : float
            Minimum z-scored composite gain to accept a merge (Eq. 10).
        w_sim : float
            Weight for similarity in the merge score.
        w_dti : float
            Weight for DTI complement (stability) in the merge score.
        w_boundary : float
            Weight for boundary strength (subtractive) in the merge score.
        target_n_domains : int or None
            Stop merging when this many clusters remain.
        boundary_n_neighbors : int
            Spatial neighbours for domain adjacency.
        min_sim_quantile : float
            Adaptive edge-pruning quantile for coarse-domain graph.
        coarse_resolution : float
            Leiden resolution for coarse-domain discovery.

        Returns
        -------
        adata : AnnData
            Modified in-place with ``obs["domain"]``, ``obs["domain_coarse"]``,
            DTI/TS columns, and similarity/adjacency entries in ``uns``.
        """
        p = self._merge(self.defaults_stage3, kwargs)

        if self.verbose:
            print("=" * 60)
            print("  Stage 3 — Transition-aware hierarchical refinement")
            print("=" * 60)

        # ---- 1) DTI & soft assignment (Eqs. 7-9) ----
        self._log("  Computing transition scores and DTI ...")
        utils.compute_transition_scores_entropy(
            adata,
            domain_key=p["domain_key"],
        )

        # ---- 2) domain similarity ----
        self._log("  Building inter-domain similarity matrix ...")
        utils.compute_domain_sim(
            adata,
            domain_key=p["domain_key"],
        )

        # ---- 3) spatial adjacency & boundary strength ----
        self._log("  Computing domain adjacency and boundary strength ...")
        utils.compute_domain_adjacency(
            adata,
            domain_key=p["domain_key"],
            n_neighbors=p["boundary_n_neighbors"],
        )

        # ---- 4) DTI-guided hierarchical merging (Eq. 10) ----
        self._log("  Performing DTI-guided domain merging ...")
        utils.merge_clusters_by_DTI(
            adata,
            domain_key=p["domain_key"],
            new_domain_key=p["new_domain_key"],
            min_gain=p["min_gain"],
            max_dti_threshold=p["max_dti_threshold"],
            w_sim=p["w_sim"],
            w_dti=p["w_dti"],
            w_boundary=p["w_boundary"],
            target_n_domains=p["target_n_domains"],
        )

        # ---- 5) coarse-domain discovery ----
        self._log("  Discovering coarse domains ...")
        utils.search_coarse_domains(
            adata,
            domain_key=p["new_domain_key"],
            min_sim_quantile=p["min_sim_quantile"],
            resolution=p["coarse_resolution"],
        )

        if self.verbose:
            n_domain = adata.obs[p["new_domain_key"]].nunique()
            n_coarse = adata.obs[f"{p['new_domain_key']}_coarse"].nunique()
            print(f"  Stage 3 complete — {n_domain} domains, {n_coarse} coarse domains.\n")

        return adata

    # ------------------------------------------------------------------
    # Stage 4 — Tree-guided multiscale gene attribution
    # ------------------------------------------------------------------

    def run_stage4(self, adata: sc.AnnData, **kwargs) -> dict:
        """Tree-guided multiscale gene attribution.

        Constructs node-split tasks from the three-level hierarchy, optimises a
        coefficient matrix under tree-group lasso and exclusive-lasso penalties,
        and returns per-edge / per-node / global gene rankings (Eqs. 14-18).

        Parameters
        ----------
        adata : AnnData
            Must contain ``obs["fine"]``, ``uns["domain_map"]``,
            ``uns["domain_coarse_map"]``, and a log-normalised ``.X``.
        max_genes : int or None
            Number of highly variable genes to retain.
        lam : float
            Tree-group lasso regularisation strength.
        lam_excl : float
            Exclusive-lasso regularisation strength.
        max_iter : int
            Maximum proximal-gradient iterations.
        tol : float
            Convergence tolerance on relative objective change.
        step_size : float or None
            Gradient step size (auto-estimated if None).
        top_n : int
            Number of top genes in rankings.
        random_state : int
            Random seed.
        use_tree : bool
            If False, use flat-mode (single task group).

        Returns
        -------
        gene_tree : dict
            Keys: ``global``, ``node_rank``, ``edge_rank``, ``edge_rank_bc``,
            ``W``, ``task_info``, ``groups``, ``lambdas``, ``history``.
        """
        p = self._merge(self.defaults_stage4, kwargs)

        if self.verbose:
            print("=" * 60)
            print("  Stage 4 — Tree-guided multiscale gene attribution")
            print("=" * 60)
            print(f"  Optimising {p['max_genes']} genes, lam={p['lam']}, "
                  f"lam_excl={p['lam_excl']}, max_iter={p['max_iter']} ...")

        out_dir = os.path.join(self.output_dir, "stage4_gene_tree")

        gene_tree = TGL_module.rank_multiscale_genes_tree(
            adata,
            out_dir=out_dir,
            max_genes=p["max_genes"],
            lam=p["lam"],
            lam_excl=p["lam_excl"],
            max_iter=p["max_iter"],
            tol=p["tol"],
            step_size=p["step_size"],
            top_n=p["top_n"],
            random_state=p["random_state"],
            verbose=p.get("verbose", self.verbose),
            use_tree=p["use_tree"],
        )

        self.gene_tree = gene_tree

        if self.verbose:
            n_edges = len(gene_tree.get("edge_rank", {}))
            print(f"  Stage 4 complete — {n_edges} edge programs, "
                  f"gene_tree stored in self.gene_tree.\n")

        return gene_tree

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        adata: sc.AnnData,
        stages: Tuple[int, ...] = (1, 2, 3, 4),
        stage1_kwargs: Optional[Dict[str, Any]] = None,
        stage2_kwargs: Optional[Dict[str, Any]] = None,
        stage3_kwargs: Optional[Dict[str, Any]] = None,
        stage4_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full stEDGE pipeline (or a subset of stages).

        Parameters
        ----------
        adata : AnnData
            Input data with ``.X``, ``obsm["X_pca"]``, ``obsm["spatial"]``.
        stages : tuple
            Which stages to run, e.g. ``(1, 2)`` for fine domains only.
        stage1_kwargs : dict or None
            Extra parameters for Stage 1.
        stage2_kwargs : dict or None
            Extra parameters for Stage 2.
        stage3_kwargs : dict or None
            Extra parameters for Stage 3.
        stage4_kwargs : dict or None
            Extra parameters for Stage 4.

        Returns
        -------
        results : dict
            ``{"adata": adata, "gene_tree": gene_tree_or_None}``.
        """
        if self.verbose:
            print("=" * 60)
            print("  stEDGE — Full pipeline")
            print("=" * 60)

        gene_tree = None
        s1, s2, s3, s4 = (stage1_kwargs or {}), (stage2_kwargs or {}), \
                          (stage3_kwargs or {}), (stage4_kwargs or {})

        if 1 in stages:
            self.run_stage1(adata, **s1)
        if 2 in stages:
            self.run_stage2(adata, **s2)
        if 3 in stages:
            self.run_stage3(adata, **s3)
        if 4 in stages:
            gene_tree = self.run_stage4(adata, **s4)

        if self.verbose:
            print("=" * 60)
            print("  stEDGE pipeline complete.")
            print("=" * 60)

        return {"adata": adata, "gene_tree": gene_tree}

    # ------------------------------------------------------------------
    # Parameter presets
    # ------------------------------------------------------------------

    def set_defaults(self, stage: int, **kwargs) -> None:
        """Update the default parameters for a given stage.

        Parameters
        ----------
        stage : int
            Stage number (1-4).
        **kwargs
            Parameter overrides for that stage.
        """
        target = {
            1: self.defaults_stage1,
            2: self.defaults_stage2,
            3: self.defaults_stage3,
            4: self.defaults_stage4,
        }.get(stage)
        if target is None:
            raise ValueError(f"stage must be 1-4, got {stage}")
        target.update(kwargs)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_stage1(self, adata: sc.AnnData, spot_size: float = 100, **kwargs):
        """Plot Stage 1 diagnostics: cluster count, p_edge, p_edge histogram."""
        plot.plot_cluster_count_resolution(
            adata, output_dir=self.output_dir, **kwargs,
        )
        plot.plot_spatial(
            adata, colors=["p_edge", "a_p_edge"],
            spot_size=spot_size, cmap="inferno",
            output_dir=self.output_dir, **kwargs,
        )
        plot.plot_p_edge_distribution(
            adata, output_dir=self.output_dir, **kwargs,
        )

    def plot_stage2(self, adata: sc.AnnData, spot_size: float = 100, **kwargs):
        """Plot Stage 2 diagnostics: seeds, region_label, gradient, fine."""
        for key, legend in [("labeled_seeds", "right"),
                            ("region_label", "right margin"),
                            ("con_region", "right margin")]:
            if key in adata.obs:
                plot.plot_spatial(
                    adata, colors=[key], spot_size=spot_size,
                    legend_loc=legend if legend != "right" else None,
                    colorbar_loc=legend if legend == "right" else None,
                    output_dir=self.output_dir, **kwargs,
                )
        if "grad_p_edge" in adata.obsm:
            plot.plot_gradient_field(adata, output_dir=self.output_dir, **kwargs)
            plot.plot_gradient_magnitude(adata, output_dir=self.output_dir, **kwargs)
        if "fine" in adata.obs:
            plot.plot_spatial(
                adata, colors=["fine"], spot_size=spot_size,
                colorbar_loc=None, legend_loc="right margin",
                output_dir=self.output_dir, **kwargs,
            )

    def plot_stage3(self, adata: sc.AnnData, spot_size: float = 100, **kwargs):
        """Plot Stage 3 diagnostics: DTI, domain/coarse spatial, graph, tree."""
        # DTI / TS spatial
        for prefix in ["fine", "domain"]:
            for suffix in ["_DTI", "_TS"]:
                col = f"{prefix}{suffix}"
                if col in adata.obs:
                    plot.plot_spatial(
                        adata, colors=[col], spot_size=spot_size,
                        cmap="inferno", output_dir=self.output_dir, **kwargs,
                    )
        # Domain and coarse spatial
        for key in ["domain", "domain_coarse"]:
            if key in adata.obs:
                plot.plot_spatial(
                    adata, colors=[key], spot_size=spot_size,
                    colorbar_loc=None, legend_loc="right margin",
                    output_dir=self.output_dir, **kwargs,
                )
        # Domain graph and tree
        if "domain" in adata.obs:
            plot.plot_domain_graph(
                adata, domain_key="domain", color_mode="dti",
                output_dir=self.output_dir, **kwargs,
            )
            plot.plot_domain_tree(
                adata, output_dir=self.output_dir, **kwargs,
            )

    def plot_stage4(self, adata: sc.AnnData, gene_tree: dict = None, **kwargs):
        """Plot Stage 4 diagnostics: global gene bar chart, spatial module score."""
        if gene_tree is None:
            gene_tree = getattr(self, "gene_tree", None)
        if gene_tree is None:
            raise ValueError("gene_tree not provided and not found in self.gene_tree.")
        from . import TGL_module
        # Global gene bar chart is lightweight matplotlib; spatial score
        # requires TGL_module.  Pass through for manual control.
        TGL_module.batch_plot_gene_tree_results(
            adata, gene_tree, base_out_dir=self.output_dir, **kwargs,
        )

    def plot_all(self, adata: sc.AnnData, spot_size: float = 100, **kwargs):
        """Run all plotting (Stage 1 + 2 + 3)."""
        self.plot_stage1(adata, spot_size=spot_size, show=False, **kwargs)
        self.plot_stage2(adata, spot_size=spot_size, show=False, **kwargs)
        self.plot_stage3(adata, spot_size=spot_size, show=False, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print a message if verbose mode is on."""
        if self.verbose:
            print(msg)
