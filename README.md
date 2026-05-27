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

### Step 2. Install dependency packages

```
pip install -r requirements.txt
```

### Step 3. Install stEDGE

For local installation, we recommend editable mode:

```bash
pip install -e .
```

Alternatively, you can install `stEDGE` using the setup script:

```bash
pip install setuptools==58.2.0
python setup.py build
python setup.py install
```

The environment configuration is then completed.

## Tutorials

For installation instructions and step-by-step tutorials, please refer to the online documentation:

https://stedge-tutorials.readthedocs.io/en/latest/

Tutorial source repository:

https://github.com/yihe-csu/stEDGE_Tutorials

## Citation

If you use `stEDGE` in your work, please cite:

He, Y. **stEDGE enables edge-guided multiscale reconstruction of hierarchical spatial domains and transition interfaces in spatial transcriptomics**. 2026.

Citation information will be updated upon publication.