"""
Parkinson's disease action recognition DataLoader.

Supports pre-split train/test directories and stratified K-Fold (sample-level splits).
No on-the-fly augmentation: loads skeleton `.npy` from disk and optional HHT sidecars.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Tuple, List, Dict, Optional
import random


def _align_hht_time(hht: np.ndarray, target_T: int) -> np.ndarray:
    """Linearly interpolate HHT time axis to `target_T` to match skeleton length."""
    T, F = hht.shape
    if T == target_T:
        return hht.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_T, dtype=np.float64)
    out = np.zeros((target_T, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(x_new, x_old, hht[:, f].astype(np.float64)).astype(np.float32)
    return out


class ParkinsonDataset(Dataset):
    """
    Data: `(T, V, 3)`; `T` must match `model.num_frame` in config.
    Labels: integers in `labels/<stem>.txt`, one per skeleton file.
    """

    def __init__(
        self,
        data_dir: str,
        action_type: str,
        samples: List[Dict],
        use_hht_features: bool = False,
        hht_subdir: str = "hht",
        hht_stem_suffix: str = "_hht",
        hht_feature_dim: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.action_type = action_type
        self.samples = samples
        self.use_hht_features = use_hht_features
        self.hht_subdir = hht_subdir
        self.hht_stem_suffix = hht_stem_suffix
        self.hht_feature_dim = hht_feature_dim
        self._print_label_info()

    def _hht_npy_path(self, skeleton_npy_path: str) -> Path:
        p = Path(skeleton_npy_path)
        return p.parent / self.hht_subdir / f"{p.stem}{self.hht_stem_suffix}.npy"

    def _print_label_info(self):
        labels = [s["label"] for s in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"[OK] Loaded {len(self.samples)} samples")
        print(f"   Label distribution: {dict(zip(unique_labels, counts))}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        sample = self.samples[idx]
        data = np.load(sample["npy_path"]).astype(np.float32)
        label = sample["label"]
        patient_id = sample["patient_id"]

        if self.use_hht_features:
            hp = self._hht_npy_path(sample["npy_path"])
            if not hp.is_file():
                raise FileNotFoundError(
                    f"HHT feature file not found: {hp}\n"
                    f"(skeleton: {sample['npy_path']}, expected shape (T, {self.hht_feature_dim}))"
                )
            hht = np.load(hp).astype(np.float32)
            if hht.ndim != 2 or hht.shape[1] != self.hht_feature_dim:
                raise ValueError(
                    f"HHT array at {hp} has shape {hht.shape}, expected (*, {self.hht_feature_dim})"
                )
            hht = _align_hht_time(hht, data.shape[0])
        else:
            hht = np.zeros((data.shape[0], self.hht_feature_dim), dtype=np.float32)

        if hht.shape[0] != data.shape[0]:
            hht = _align_hht_time(hht, data.shape[0])

        return (
            torch.from_numpy(data).float(),
            torch.from_numpy(hht).float(),
            label,
            patient_id,
        )


class ParkinsonDataModule:
    def __init__(
        self,
        data_dir: str = "./data",
        action_type: str = "LA",
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        use_split: bool = False,
        split_dir: Optional[str] = None,
        use_hht_features: bool = False,
        hht_subdir: str = "hht",
        hht_stem_suffix: str = "_hht",
        hht_feature_dim: int = 10,
    ):
        self.data_dir = data_dir
        self.action_type = action_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_split = use_split
        self.split_dir = split_dir
        self.use_hht_features = use_hht_features
        self.hht_subdir = hht_subdir
        self.hht_stem_suffix = hht_stem_suffix
        self.hht_feature_dim = hht_feature_dim

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print("=" * 80)
        print("ParkinsonDataModule Initialized")
        print("=" * 80)

        if use_split and split_dir:
            print(f"[OK] Using pre-split data from: {split_dir}")
            self.train_samples = self._load_samples_from_dir(Path(split_dir) / "train")
            self.test_samples = self._load_samples_from_dir(Path(split_dir) / "test")
            print(f"Train samples: {len(self.train_samples)}")
            print(f"Test samples: {len(self.test_samples)}")
            print(f"Train patients: {len(set(s['patient_id'] for s in self.train_samples))}")
            print(f"Test patients: {len(set(s['patient_id'] for s in self.test_samples))}")
        else:
            print(f"[OK] Using K-Fold mode from: {data_dir}")
            self.all_samples = self._load_all_samples()
            print(f"Total samples: {len(self.all_samples)}")
            print(f"Total patients: {len(set(s['patient_id'] for s in self.all_samples))}")

        print(f"Batch size: {batch_size}")
        print(f"Num workers: {num_workers}")
        print(f"Random seed: {seed}")
        if use_hht_features:
            print(
                f"[OK] HHT features: load from subdir '{hht_subdir}' (*{hht_stem_suffix}.npy), dim={hht_feature_dim}"
            )
        print("=" * 80)

    def _sample_dict(self, npy_file: Path, patient_id: str, side: str, label: int) -> Dict:
        return {
            "patient_id": patient_id,
            "side": side,
            "npy_path": str(npy_file),
            "label": label,
            "original_label": label,
        }

    def _load_samples_from_dir(self, data_dir: Path) -> List[Dict]:
        labels_dir = data_dir / "labels"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        samples = []
        for npy_file in sorted(data_dir.glob("*.npy")):
            filename = npy_file.stem
            if "_seg" in filename:
                base_name = filename.split("_seg")[0]
            else:
                base_name = filename
            parts = base_name.rsplit("-", 1)
            if len(parts) == 2:
                patient_id, side = parts
            else:
                patient_id, side = base_name, "unknown"
            label_file = labels_dir / f"{filename}.txt"
            if not label_file.exists():
                print(f"[WARN] Label not found for {filename}")
                continue
            with open(label_file, "r") as f:
                label = int(f.read().strip())
            samples.append(self._sample_dict(npy_file, patient_id, side, label))
        return samples

    def _load_all_samples(self) -> List[Dict]:
        action_dir = Path(self.data_dir)
        labels_dir = action_dir / "labels"
        samples = []
        for npy_file in sorted(action_dir.glob("*.npy")):
            filename = npy_file.stem
            if "_seg" in filename:
                base_name = filename.split("_seg")[0]
            else:
                base_name = filename
            parts = base_name.rsplit("-", 1)
            if len(parts) == 2:
                patient_id, side = parts
            else:
                patient_id, side = base_name, "unknown"
            label_file = labels_dir / f"{filename}.txt"
            if not label_file.exists():
                continue
            with open(label_file, "r") as f:
                label = int(f.read().strip())
            samples.append(self._sample_dict(npy_file, patient_id, side, label))
        return samples

    def get_train_test_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        if not self.use_split:
            raise ValueError("use_split=False, please use get_fold_dataloaders() instead")

        print(f"\n{'=' * 80}\nLoading Pre-Split Train/Test Data\n{'=' * 80}")
        print(f"Train samples: {len(self.train_samples)}")
        print(f"Test samples: {len(self.test_samples)}")

        ds_kw = dict(
            use_hht_features=self.use_hht_features,
            hht_subdir=self.hht_subdir,
            hht_stem_suffix=self.hht_stem_suffix,
            hht_feature_dim=self.hht_feature_dim,
        )
        train_dataset = ParkinsonDataset(self.data_dir, self.action_type, self.train_samples, **ds_kw)
        test_dataset = ParkinsonDataset(self.data_dir, self.action_type, self.test_samples, **ds_kw)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._print_label_distribution(train_dataset, test_dataset)
        print(f"{'=' * 80}\n")
        return train_loader, test_loader

    def get_stratified_k_fold_splits(self, n_splits: int = 5) -> List[Tuple[List[Dict], List[Dict]]]:
        if self.use_split:
            raise ValueError("use_split=True, K-Fold is not available")

        sample_labels = np.array([s["label"] for s in self.all_samples])
        print(f"\n{'=' * 80}\nStratified K-Fold Cross-Validation (Sample-Level)\n{'=' * 80}")
        print(f"Total samples: {len(self.all_samples)}")
        print("Label distribution:")
        unique_labels, counts = np.unique(sample_labels, return_counts=True)
        for lab, cnt in zip(unique_labels, counts):
            print(f"  Label {lab}: {cnt:3d} samples ({cnt / len(sample_labels) * 100:5.2f}%)")
        print(f"{'=' * 80}\n")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(np.arange(len(self.all_samples)), sample_labels)
        ):
            train_samples = [self.all_samples[i] for i in train_idx]
            test_samples = [self.all_samples[i] for i in test_idx]
            train_labels = sample_labels[train_idx]
            test_labels = sample_labels[test_idx]
            print(f"Fold {fold_idx}:")
            print(f"  Train: {len(train_samples)} samples, labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
            print(f"  Test:  {len(test_samples)} samples, labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
            splits.append((train_samples, test_samples))
        print(f"\n{'=' * 80}\n")
        return splits

    def get_fold_dataloaders(
        self,
        fold: int,
        n_splits: int = 5,
        use_stratified: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        assert 0 <= fold < n_splits, f"fold must be in [0, {n_splits - 1}]"
        if self.use_split:
            raise ValueError("use_split=True, please use get_train_test_dataloaders() instead")

        if use_stratified:
            splits = self.get_stratified_k_fold_splits(n_splits)
            train_samples, test_samples = splits[fold]
        else:
            idx = np.arange(len(self.all_samples))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            splits_list = list(kf.split(idx))
            train_idx, test_idx = splits_list[fold]
            train_samples = [self.all_samples[i] for i in train_idx]
            test_samples = [self.all_samples[i] for i in test_idx]
            print(f"\n{'=' * 80}\nK-Fold (non-stratified) fold {fold}\n{'=' * 80}")

        print(f"\n{'=' * 80}\nFold {fold}/{n_splits - 1}\n{'=' * 80}")
        print(f"Train samples: {len(train_samples)}")
        print(f"Test samples: {len(test_samples)}")

        ds_kw = dict(
            use_hht_features=self.use_hht_features,
            hht_subdir=self.hht_subdir,
            hht_stem_suffix=self.hht_stem_suffix,
            hht_feature_dim=self.hht_feature_dim,
        )
        train_dataset = ParkinsonDataset(self.data_dir, self.action_type, train_samples, **ds_kw)
        test_dataset = ParkinsonDataset(self.data_dir, self.action_type, test_samples, **ds_kw)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._print_label_distribution(train_dataset, test_dataset)
        print(f"{'=' * 80}\n")
        return train_loader, test_loader

    def _print_label_distribution(self, train_dataset, test_dataset):
        train_labels = [s["label"] for s in train_dataset.samples]
        test_labels = [s["label"] for s in test_dataset.samples]
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        print("\n[stats] Label distribution (train / test):")
        print(f"  Train: {dict(zip(train_unique, train_counts))}")
        tt = int(train_counts.sum())
        train_pct = {int(lab): f"{cnt / tt * 100:.1f}%" for lab, cnt in zip(train_unique, train_counts)}
        print(f"         {train_pct}")
        print(f"  Test:  {dict(zip(test_unique, test_counts))}")
        te = int(test_counts.sum())
        test_pct = {int(lab): f"{cnt / te * 100:.1f}%" for lab, cnt in zip(test_unique, test_counts)}
        print(f"         {test_pct}")