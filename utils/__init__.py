"""HHT helpers: CEEMDAN + Hilbert IA/IF; skeleton -> 1D trajectory -> (T, 2K) feature matrix."""

from .hht_analyzer import HHTAnalyzer, ia_if_feature_matrix, filter_if_physical
from .skeleton_to_signal import (
    ft_finger_tip_distance,
    la_left_knee_y,
    joint_channel_series,
)

__all__ = [
    "HHTAnalyzer",
    "ia_if_feature_matrix",
    "filter_if_physical",
    "ft_finger_tip_distance",
    "la_left_knee_y",
    "joint_channel_series",
]
