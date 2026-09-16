# stEDGE Demo

This directory demonstrates the four-stage **stEDGE** mouse embryo workflow for reviewer testing and software validation. The script follows `Tutorial_1_Mouse_embryo_RunAll.ipynb` and saves only its six key figures.

## Demo dataset

Download the processed mouse embryo spatial transcriptomics dataset `E9.5_E1S1.h5ad` from [Zenodo](https://doi.org/10.5281/zenodo.22781985) and place it at:

```text
demo/data/E9.5_E1S1.h5ad
```

The section is derived from the publicly available [MOSTA atlas](https://db.cngb.org/stomics/mosta/). The `.h5ad` file is hosted separately and is not committed to this Git repository.

## Installation

From the root of the stEDGE repository, install the package and its dependencies:

```bash
pip install -e .
```

Alternatively, install the published package:

```bash
pip install stEDGE
```

## Run the demo

From the repository root:

```bash
python demo/run_demo.py
```

The default input and output paths are resolved relative to `run_demo.py`, so calling the script by its absolute path also works from another directory. To use other locations:

```bash
python demo/run_demo.py --data /path/to/E9.5_E1S1.h5ad --output /path/to/output
```

The script filters and normalizes the count matrix, computes highly variable genes and PCA, and runs stages 1–4 in sequence. It does not require interactive notebook cells or a GPU.

## Output

Successful runs write these six images directly under `demo/output/`:

- `boundary_and_domain_DTI.png`: boundary probability and domain DTI
- `multiscale_domains.png`: fine, domain, and coarse spatial labels
- `domain_graph.png`: DTI and cluster graph views
- `Domain_Hierarchy_Tree_DTI.png`: three-level hierarchy
- `spatial_module_score_domain_11.png`: domain 11 and spatial gene-module score
- `gradient_bar_domain_11.png`: top gene weights for domain 11

The script does not run the notebook's Evaluation or Save results sections. It does not export AnnData, labels, embeddings, gene rankings, logs, or summary files. Runtime and inferred domain counts are printed in the terminal.

## Runtime

| Test condition | Value |
| --- | --- |
| CPU | Intel Core i7-12700H; 14 physical cores, 20 logical processors |
| System memory | 34.1 GB |
| Operating system | Windows 11, build 26200 |
| Python | 3.12.5 |
| stEDGE / Scanpy | 1.1.2 / 1.10.2 |
| Measured elapsed time | **104.57 seconds** |

## Directory structure

```text
demo/
├── README.md
├── run_demo.py
├── data/
│   └── E9.5_E1S1.h5ad   # downloaded from Zenodo
└── output/              # created by the script
```

## Full tutorials

Full real-data tutorials covering Stereo-seq, Visium, Slide-seqV2, and Xenium datasets are available in [stEDGE Tutorials](https://github.com/yihe-csu/stEDGE_Tutorials), with [online documentation](https://stedge-tutorials.readthedocs.io/en/latest/).
