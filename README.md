# **stEDGE: Edge-guided multiscale reconstruction of hierarchical spatial domains and transition interfaces in spatial transcriptomics**

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

### Step 2. Install dependency packages

```
pip install -r requirements.txt
```

### Step 3. Install stEDGE

```
pip install setuptools==58.2.0
python setup.py build
python setup.py install
```

Alternatively, for development mode:

```
pip install -e .
```

The environment configuration is then completed.

## Tutorials

Tutorials are provided in the documentation folder and cover representative applications of `stEDGE`.

- **Tutorial 1:** step-by-step analysis of mouse embryo MOSTA data.
- **Tutorial 2:** transition-aware reconstruction of Crohn’s disease Visium data.
- **Tutorial 3:** hierarchical reconstruction of human breast cancer Visium data.
- **Tutorial 4:** layer-based reconstruction of mouse cerebellum Slide-seqV2 data.
- **Tutorial 5:** cell-resolved reconstruction of human fibrotic lung Xenium data.

## Citation

If you use `stEDGE` in your work, please cite:

He, Y. **stEDGE enables edge-guided multiscale reconstruction of hierarchical spatial domains and transition interfaces in spatial transcriptomics**. 2026.

Citation information will be updated upon publication.