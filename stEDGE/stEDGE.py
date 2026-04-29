import os
from . import utils
from . import preprocess
from . import plot
from . import TGL_module
from copy import deepcopy


class StEDGE:
    """
    Stage 1 – Fine-grained domain discovery (6-step pipeline) wrapper.

    - Main entry: `StEDGE(...).fine_grained_domain_discovery(adata)`
    - Modifies `adata` in-place.
    """

    DEFAULT_FGD_CFG = {
        "step1": dict(
            method="leiden",
            resolution_range=(0.1, 5, 0.1),
            n_neighbors=15,
            use_rep="X_pca",
            tau_ARI=0.9,
            tau_k=3,
            use_stable="core",
            refinement=False,
            cal=False,
            plot=True,
        ),
        "step2": dict(n_neighbors=7),
        "step3": dict(n_neighbors=6, p_cap_start=0.10, p_cap_step=0.02, p_cap_max=0.50),
        "step4": dict(grad_k=6, grad_norm=True, global_thr_k=0.25),
        "step5": dict(con_th=0.7, detect_unassigned=True, con_th_inner=0.7, min_size=20),
        "step6": dict(n_components=50, n_neighbors=7),
    }

     # ===== Stage 2 =====
    DEFAULT_MEG_CFG = {
        "step7": dict(domain_key="fine"),
        "step8": dict(domain_key="fine"),
        "step9": dict(domain_key="fine", n_neighbors=6),
        "step10": dict(
            domain_key="fine",
            new_domain_key="domain",
            min_gain=0.05,
            w_sim=1.0,
            w_dti=0.0,
            w_boundary=0.0,
            target_n_domains=None,
        ),
        "step11": dict(domain_key="domain", min_sim_th=0.1, resolution=1),
    }

    DEFAULT_TGL_CFG = {
        "step12": dict(
            # keys 与你的 TGL 函数默认一致（可按需改）
            max_genes=3000,
            lam=0.2,
            lam_excl=0.1,
            max_iter=200,
            tol=1e-4,
            step_size=None,
            top_n=100,
            random_state=0,
            verbose=True,
        )
    }

    def __init__(self, 
                 output_process_dir, 
                 cfg=None, 
                 make_dir=True, 
                 verbose=True):
        self.output_process_dir = output_process_dir
        self.make_dir = make_dir
        self.verbose = verbose

        self.cfg = {
            "stage1": deepcopy(self.DEFAULT_FGD_CFG),
            "stage2": deepcopy(self.DEFAULT_MEG_CFG),
            "stage3": deepcopy(self.DEFAULT_TGL_CFG), 
        }
        if cfg:
            self.update_cfg(cfg)

        if self.make_dir:
            os.makedirs(self.output_process_dir, exist_ok=True)

    def update_cfg(self, cfg_update: dict):
        """
        Update config (shallow merge).

        cfg_update example:
        {
          "stage1": {"step3": {"p_cap_max": 0.6}},
          "stage2": {"step10": {"min_gain": 0.1}}
        }
        """
        for stage_name, stage_cfg in cfg_update.items():
            if stage_name not in self.cfg:
                raise KeyError(f"Unknown stage: {stage_name}. Allowed: {list(self.cfg.keys())}")
            if not isinstance(stage_cfg, dict):
                raise TypeError(f"{stage_name} config must be a dict.")
            for step_name, params in stage_cfg.items():
                if step_name not in self.cfg[stage_name]:
                    raise KeyError(
                        f"Unknown step: {stage_name}.{step_name}. "
                        f"Allowed: {list(self.cfg[stage_name].keys())}"
                    )
                if not isinstance(params, dict):
                    raise TypeError(f"{stage_name}.{step_name} config must be a dict.")
                self.cfg[stage_name][step_name].update(params)

    def _merged_cfg(self, cfg_override=None):
        """Get effective config (self.cfg overridden by cfg_override)."""
        merged = deepcopy(self.cfg)
        if cfg_override:
            for stage_name, stage_cfg in cfg_override.items():
                if stage_name not in merged:
                    raise KeyError(f"Unknown stage: {stage_name}. Allowed: {list(merged.keys())}")
                if not isinstance(stage_cfg, dict):
                    raise TypeError(f"{stage_name} override must be a dict.")
                for step_name, params in stage_cfg.items():
                    if step_name not in merged[stage_name]:
                        raise KeyError(
                            f"Unknown step: {stage_name}.{step_name}. "
                            f"Allowed: {list(merged[stage_name].keys())}"
                        )
                    if not isinstance(params, dict):
                        raise TypeError(f"{stage_name}.{step_name} override must be a dict.")
                    merged[stage_name][step_name].update(params)
        return merged
    

    # ==================== Stage 1 ====================
    def fine_grained_domain_discovery(self, adata, cfg_override=None):
        """
        Run Stage 1 pipeline (always runs all 6 steps).

        Returns
        -------
        adata : AnnData
        merged_cfg : dict
            Effective config used in this run.
        """
        merged_cfg = self._merged_cfg(cfg_override)
        stage1 = merged_cfg["stage1"]

        if self.make_dir:
            os.makedirs(self.output_process_dir, exist_ok=True)

        if self.verbose:
            print("---------------- # Step 1/6 – Consensus pre-clustering ---------------------")
        preprocess.consensus_clustering(adata, **stage1["step1"])


        if self.verbose:
            print("---------------- # Step 2/6: compute boundary probability ---------------------")
        utils.compute_boundary_probability(adata, **stage1["step2"])

        if self.verbose:
            print("---------------- # Step 3/6 – Seed region detection  ---------------------")
        utils.find_seed_regions(adata, **stage1["step3"])
        if self.verbose:
            print("---------------- # Step 4/6 – Region growing segmentation ---------------------")
        utils.compute_gradient(
            adata,
            k=stage1["step4"]["grad_k"],
            norm=stage1["step4"]["grad_norm"],
        )
        utils.region_growing_segmentation(
            adata,
            global_thr_k=stage1["step4"]["global_thr_k"],
            output_dir=self.output_process_dir,
        )

        if self.verbose:
            print("---------------- # Step 5/6 – Consensus-based domain propagation---------------------")
        utils.propagate_domains_from_regions(adata, **stage1["step5"])

        if self.verbose:
            print("---------------- # Step 6/6 – PCA-based label propagation---------------------")
        utils.label_propagation_with_pca(adata, **stage1["step6"])
        if self.verbose:
            print(">>> Stage 1 – Fine-grained domain discovery completed!")

        return adata, merged_cfg
  

    def plot_stage1_fig(self, adata, spot_size, **kwargs):
        """
        Stage1 结果图统一绘制入口（封装为一个 plot 函数）。
        默认输出到 self.output_process_dir。
        """

        return plot.plot_stage1_fig(
            adata=adata,
            output_dir=self.output_process_dir,
            spot_size=spot_size,
            **kwargs,
        )
    
    def plot_stage2_fig(self, adata, spot_size, **kwargs):
        """
    Stage2 结果图统一绘制入口（封装为一个 plot 函数）。
    默认输出到 self.output_process_dir。
        """

        return plot.plot_stage2_fig(
            adata=adata,
            output_dir=self.output_process_dir,
            spot_size=spot_size,
            **kwargs,
        )
    

    def plot(self, adata, spot_size, **kwargs):
        """
    Stage2 结果图统一绘制入口（封装为一个 plot 函数）。
    默认输出到 self.output_process_dir。
        """
        plot.plot_stage1_fig(
            adata=adata,
            output_dir=self.output_process_dir,
            spot_size=spot_size,
            **kwargs,
        )

        plot.plot_stage2_fig(
                    adata=adata,
                    output_dir=self.output_process_dir,
                    spot_size=spot_size,
                    **kwargs,
                )
        return 

    # ==================== Stage 2====================
    def domain_evaluation_and_hierarchical_refinement(self, adata, cfg_override=None):
        """
        Stage 2 – Domain evaluation and hierarchical refinement (Steps 7-11).

        Returns
        -------
        adata : AnnData
        df_dti : pandas.DataFrame
            Output of compute_transition_scores_entropy.
        meta_map : Any
            Output of merge_clusters_by_DTI.
        merged_cfg : dict
            Effective config used in this run.
        """
        merged_cfg = self._merged_cfg(cfg_override)
        stage2 = merged_cfg["stage2"]

        if self.verbose:
            print("---------------- # Step 7/11 – Transition scoring (TS & DTI entropy) -------")
        utils.compute_transition_scores_entropy(adata, **stage2["step7"])

        if self.verbose:
            print("---------------- # Step 8/11 – Inter-domain similarity graph construction --")
        utils.compute_domain_sim(adata, **stage2["step8"])

        if self.verbose:
            print("---------------- # Step 9/11 – Spatial adjacency & boundary strength -------")
        utils.compute_domain_adjacency(adata, **stage2["step9"])

        if self.verbose:
            print("---------------- # Step 10/11 – DTI-guided region merging ------------------")
        meta_map = utils.merge_clusters_by_DTI(adata, **stage2["step10"])

        if self.verbose:
            print("---------------- # Step 11/11 – Coarse domain discovery --------------------")
        utils.search_coarse_domains(adata, **stage2["step11"])

        if self.verbose:
            print(">>> Stage 2 – Domain evaluation and hierarchical refinement completed!")

        return adata, merged_cfg
    

    def multiscale_gene_tree_discovery(self, adata, cfg_override=None, out_dir=None):
        """
        Stage 3 – Tree-guided multiscale discriminative gene discovery (Step 12).

        执行：
          gene_tree = TGL_module.rank_multiscale_genes_tree_guided_node_split_exclusive(...)

        Returns
        -------
        gene_tree : dict
            TGL_module.rank_multiscale... 的输出 out（你前面一直叫 gene_tree）
        merged_cfg : dict
            本次 run 的有效配置
        """
        merged_cfg = self._merged_cfg(cfg_override)
        stage3 = merged_cfg["stage3"]

        # 默认输出目录：{output_process_dir}/stage3_gene_tree
        if out_dir:
            out_dir = os.path.join(self.output_process_dir, "stage3_gene_tree")
            os.makedirs(out_dir, exist_ok=True)

        p = stage3["step12"]

        if self.verbose:
            print("---------------- # Step 12 – Tree-guided multiscale gene discovery ---------")

        # 2) 跑 TGL
        gene_tree = TGL_module.rank_multiscale_genes_tree(
            adata,
            out_dir=out_dir,          # ✅ Stage3 统一管理 out_dir
            **p,
        )

        # 3) 方便后续调用（可选）
        self.gene_tree = gene_tree

        if self.verbose:
            print(">>> Stage 3 – Multiscale gene discovery completed!")

        return gene_tree, merged_cfg
    
    
