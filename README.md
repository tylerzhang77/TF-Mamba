# Parkinson Action Assessment (GraphMamba) — Core Code

> **Status:** This repository is still being curated. Layout, docs, and tooling may change; a fuller release will follow—please check back later.

Training entry, data loading, model, skeleton graphs, and configuration only. No offline scalar-feature pipelines or statistical plotting scripts.

## Layout


| Path                  | Role                                                            |
| --------------------- | --------------------------------------------------------------- |
| `main.py`             | Training: `configs/config.yaml`, K-Fold or pre-split train/test |
| `model/`              | GraphMamba, Mamba blocks, optional contrastive modules          |
| `dataloader/`         | `ParkinsonDataset` / `ParkinsonDataModule`                      |
| `graph/`              | Hand (21) / HybrIK (24) adjacency definitions                   |
| `configs/config.yaml` | Default config (edit paths and hyperparameters locally)         |
| `utils/`              | Optional HHT export: CEEMDAN + Hilbert → `hht/*_hht.npy`        |


## Environment

```bash
pip install -r requirements.txt
```

Install a **PyTorch** build that matches your CUDA version from [pytorch.org](https://pytorch.org).

## Data

Set `data_dir`, `split_dir`, etc. in `configs/config.yaml`. Skeleton tensors must be `(T, V, 3)` with `T == model.num_frame` (default 240). Labels live under `labels/<stem>.txt` (single integer per file).

Optional HHT sidecars: enable `use_hht_features` / `use_hht_injection` consistently with `hht_feature_dim` and `hht_in_channels`.

From the repo root, generate HHT files compatible with the loader:

```bash
python -m utils.export_hht_npy --data_dir /path/to/skeletons --mode ft
python -m utils.export_hht_npy --data_dir /path/to/split/train --mode la --overwrite
```

`--mode ft` uses thumb–index tip distance; `--mode la` uses left-knee Y. Outputs `hht/<stem>_hht.npy`, shape `(T, 10)` (first five IMFs: IA and IF interleaved).

Placeholder dirs `data/`, `checkpoint/`, `output/` hold `.gitkeep` only; see `.gitignore` for what is ignored.

## Train

```bash
python main.py --config configs/config.yaml --gpu 0
```

## License / data ethics

Code is for research. Do not publish identifiable clinical data in public repositories.
