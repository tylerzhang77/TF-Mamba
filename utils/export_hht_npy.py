#!/usr/bin/env python3
"""
Batch-generate HHT feature `.npy` files for skeleton `.npy`, paths match the dataloader:
  <skeleton_dir>/hht/<stem>_hht.npy
Shape (T, 2*K): IA and IF interleaved for the first K IMFs.

From repo root:
  python -m utils.export_hht_npy --data_dir /path/to/data --mode ft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure imports work when run as script or python -m
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.hht_analyzer import EMDMethod, ia_if_feature_matrix
from utils.skeleton_to_signal import ft_finger_tip_distance, la_left_knee_y


def default_hht_path(skeleton_npy: Path, hht_subdir: str, stem_suffix: str) -> Path:
    return skeleton_npy.parent / hht_subdir / f"{skeleton_npy.stem}{stem_suffix}.npy"


def iter_skeleton_npys(root: Path) -> list[Path]:
    """Collect `*.npy` under `root`, or under `root/train` and `root/test` if both exist."""
    root = Path(root)
    train, test = root / "train", root / "test"
    if train.is_dir() and test.is_dir():
        files = sorted(train.glob("*.npy")) + sorted(test.glob("*.npy"))
    else:
        files = sorted(root.glob("*.npy"))
    return files


def export_one(
    npy_path: Path,
    *,
    mode: str,
    fps: float,
    n_imf: int,
    emd_method: EMDMethod,
    hht_subdir: str,
    stem_suffix: str,
    out_path: Path | None,
    overwrite: bool,
) -> Path:
    data = np.load(npy_path)
    if data.ndim != 3:
        raise ValueError(f"{npy_path}: expected (T,V,C) skeleton, got {data.shape}")

    if mode == "ft":
        sig = ft_finger_tip_distance(data)
    elif mode == "la":
        sig = la_left_knee_y(data)
    else:
        raise ValueError(f"mode must be 'ft' or 'la', got {mode}")

    feat = ia_if_feature_matrix(sig, fps=fps, n_imf=n_imf, emd_method=emd_method)
    if feat.shape[0] != data.shape[0]:
        raise RuntimeError(
            f"HHT T={feat.shape[0]} != skeleton T={data.shape[0]} for {npy_path}"
        )

    dst = out_path or default_hht_path(npy_path, hht_subdir, stem_suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return dst
    np.save(dst, feat.astype(np.float32))
    return dst


def should_skip(dst: Path, overwrite: bool) -> bool:
    return dst.exists() and not overwrite


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export HHT (IA/IF) npy for GraphMamba dataloader")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with skeleton `*.npy` files (or train/test subdirs)",
    )
    p.add_argument("--mode", type=str, choices=("ft", "la"), required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--n_imf", type=int, default=5, help="Use first K IMFs; output width is 2*K")
    p.add_argument("--emd", type=str, default="CEEMDAN", choices=("CEEMDAN", "EEMD", "EMD"))
    p.add_argument("--hht_subdir", type=str, default="hht")
    p.add_argument("--stem_suffix", type=str, default="_hht")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    files = iter_skeleton_npys(root)
    if not files:
        print(f"No .npy found under {root}", file=sys.stderr)
        return 1

    emd_method: EMDMethod = args.emd  # type: ignore[assignment]

    ok = 0
    for npy_path in files:
        dst = default_hht_path(npy_path, args.hht_subdir, args.stem_suffix)
        if args.dry_run:
            print(f"[dry-run] {npy_path} -> {dst}")
            ok += 1
            continue
        if should_skip(dst, args.overwrite):
            print(f"SKIP {npy_path.name} (exists {dst.name})")
            ok += 1
            continue
        try:
            export_one(
                npy_path,
                mode=args.mode,
                fps=args.fps,
                n_imf=args.n_imf,
                emd_method=emd_method,
                hht_subdir=args.hht_subdir,
                stem_suffix=args.stem_suffix,
                out_path=None,
                overwrite=True,
            )
            print(f"OK {npy_path.name} -> {dst}")
            ok += 1
        except Exception as e:
            print(f"FAIL {npy_path}: {e}", file=sys.stderr)

    print(f"Done {ok}/{len(files)}")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
