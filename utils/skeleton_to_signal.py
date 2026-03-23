"""
Extract 1D clinically salient trajectories from skeleton sequences (T, V, C) for HHT.
Joint indices align with calculate_feature_FT / calculate_feature_LA.
"""
from __future__ import annotations

import numpy as np

# Finger tapping: 21 hand joints, indices match graph.hand
THUMB_TIP_IDX = 4
INDEX_FINGER_TIP_IDX = 8

# Leg agility: HybrIK 24 joints in SMPL order
LEFT_KNEE_IDX = 4
Y_AXIS_IDX = 1


def ft_finger_tip_distance(
    skeleton_tvc: np.ndarray,
    thumb_idx: int = THUMB_TIP_IDX,
    index_idx: int = INDEX_FINGER_TIP_IDX,
) -> np.ndarray:
    """
    Euclidean distance between thumb tip and index finger tip over time.

    Args:
        skeleton_tvc: (T, V, C), usually xyz
    """
    s = np.asarray(skeleton_tvc, dtype=np.float32)
    thumb = s[:, thumb_idx, :]
    index = s[:, index_idx, :]
    return np.linalg.norm(thumb - index, axis=-1).astype(np.float64)


def la_left_knee_y(
    skeleton_tvc: np.ndarray,
    knee_idx: int = LEFT_KNEE_IDX,
    y_axis: int = Y_AXIS_IDX,
) -> np.ndarray:
    """Y-coordinate of the left knee joint over time."""
    s = np.asarray(skeleton_tvc, dtype=np.float32)
    return np.asarray(s[:, knee_idx, y_axis], dtype=np.float64)


def joint_channel_series(
    skeleton_tvc: np.ndarray, joint_idx: int, channel: int
) -> np.ndarray:
    """Generic: time series for one joint and axis (0=x, 1=y, 2=z)."""
    s = np.asarray(skeleton_tvc, dtype=np.float32)
    return np.asarray(s[:, joint_idx, channel], dtype=np.float64)
