# **stEDGE: Edge-guided multiscale reconstruction of hierarchical spatial domains and transition interfaces in spatial transcriptomics**

[![Documentation Status](https://readthedocs.org/projects/stedge-tutorials/badge/?version=latest)](https://stedge-tutorials.readthedocs.io/en/latest/)

<p align="center">
  <img src="./docs/Logo.png" alt="stEDGE logo" width="50%" />
</p>

## Overview

`stEDGE` is an edge-guided and interpretable framework for reconstructing multiscale tissue architecture from spatial transcriptomics data. Instead of treating tissues as flat spatial partitions, `stEDGE` explicitly models local boundary probability and domain transition intensity to identify stable compartments, transition-rich interfaces, and hierarchical spatial states.

`stEDGE` integrates four major components: consensus-based boundary modeling, edge-guided fine-domain reconstruction, transition-aware hierarchy construction, and tree-guided gene program interpretation.

![stEDGE overview](./docs/Overview.png)

Using `stEDGE`, you can:

* **Reconstruct fine-grained spatial domains** by modeling local boundary probability and growing domains from low-boundary core regions.

* **Identify transition-rich interfaces** using boundary probability, gradient signals, and domain transition indices.

* **Build multiscale tissue hierarchies** across fine, domain, and coarse spatial levels using domain graphs and hierarchy trees.

* **Interpret spatial gene programs** with tree-guided gene attribution, distinguishing shared parent-level identity from branch-specific molecular programs.

* **Analyze diverse spatial transcriptomics datasets**, including Visium, Slide-seqV2, Xenium, and other spatial omics platforms.

---

## Installation

We recommend installing `stEDGE` in a clean conda environment.

### Step 1. Create a conda environment

```bash
conda create -n stEDGE python=3.12.5
conda activate stEDGE
```

### Step 2. Install stEDGE from PyPI

```bash
pip install stEDGE
```

The installation is completed if `stEDGE` can be imported successfully:

```python
import stEDGE
print(stEDGE.__version__)
```

### Optional: Install the development version from GitHub

For local development or access to the latest source code, clone the GitHub repository and install `stEDGE` in editable mode:

```bash
git clone https://github.com/yihe-csu/stEDGE.git
cd stEDGE
pip install -e .
```

---

## Tutorials

For installation instructions and step-by-step tutorials, please refer to the online documentation:

https://stedge-tutorials.readthedocs.io/en/latest/

Tutorial source repository:

https://github.com/yihe-csu/stEDGE_Tutorials

---

## Recommended parameter settings

The following table provides platform-level parameter presets used in the current `stEDGE` tutorials. These settings are intended as recommended starting points for different spatial transcriptomics platforms and can be adjusted according to tissue complexity, spatial resolution, and the expected granularity of spatial domains.

| Platform    | Consensus interval | Seed detection                                               | Hierarchy                                        |
| ----------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| Stereo-seq  | `(1.0, 3.0, 0.1)`  | `seed_quantile=0.45`, `seed_n_neighbors=6`, `min_seed_size=5` | `min_sim_quantile=0.95`, `coarse_resolution=1.0` |
| 10x Visium  | `(0.1, 2.0, 0.1)`  | `seed_quantile=0.35`, `seed_n_neighbors=6`, `min_seed_size=0` | `min_sim_quantile=0.8`, `coarse_resolution=1.0`  |
| Slide-seqV2 | `(0.1, 1.0, 0.1)`  | `seed_quantile=0.15`, `seed_n_neighbors=6`, `min_seed_size=40` | `min_sim_quantile=0.8`, `coarse_resolution=1.0`  |
| Xenium      | `(0.1, 1.0, 0.1)`  | `seed_quantile=0.45`, `seed_n_neighbors=6`, `min_seed_size=140` | `min_sim_quantile=0.8`, `coarse_resolution=1.0`  |

For high-resolution or cell-resolved datasets, such as Slide-seqV2 and Xenium, we recommend using multiscale neighborhood representation before running the full `stEDGE` workflow.

## Citation

If you use `stEDGE` in your work, please cite:

He, Y. **stEDGE enables edge-guided multiscale reconstruction of hierarchical spatial domains and transition interfaces in spatial transcriptomics**. 2026.

Citation information will be updated upon publication.