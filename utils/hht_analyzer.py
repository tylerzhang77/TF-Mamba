"""
CEEMDAN / EMD decomposition + Hilbert transform for instantaneous amplitude (IA) and frequency (IF).
Matches the HHTAnalyzer logic used in legacy scripts calculate_feature_FT.py and calculate_feature_LA.py.
"""
from __future__ import annotations

import warnings
from typing import Literal, Tuple

import numpy as np
from PyEMD import CEEMDAN, EEMD, EMD
from scipy.signal import hilbert, medfilt, savgol_filter

warnings.filterwarnings("ignore")

EMDMethod = Literal["EMD", "EEMD", "CEEMDAN"]


class HHTAnalyzer:
    def __init__(self, emd_method: EMDMethod = "CEEMDAN", max_imf: int = 10):
        self.emd_method = emd_method
        self.max_imf = max_imf

        if emd_method == "EMD":
            self.emd = EMD()
        elif emd_method == "EEMD":
            self.emd = EEMD()
        elif emd_method == "CEEMDAN":
            self.emd = CEEMDAN()
        else:
            raise ValueError(f"Unknown EMD method: {emd_method}")

    def decompose_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64).ravel()
        try:
            if self.emd_method == "EMD":
                imfs = self.emd.emd(signal, max_imf=self.max_imf)
            elif self.emd_method == "EEMD":
                imfs = self.emd.eemd(signal, max_imf=self.max_imf)
            else:
                imfs = self.emd.ceemdan(signal, max_imf=self.max_imf)

            imfs = np.atleast_2d(imfs)
            return imfs
        except Exception as e:
            warnings.warn(f"EMD/CEEMDAN failed ({e}); using raw signal as single IMF.")
            return signal.reshape(1, -1)

    def compute_hilbert_spectrum(
        self, imfs: np.ndarray, fps: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return IA, IF (Hz), IF_normalized, IP."""
        n_imfs, T = imfs.shape
        IA = np.zeros((n_imfs, T))
        IF_normalized = np.zeros((n_imfs, T))
        IP = np.zeros((n_imfs, T))

        for i in range(n_imfs):
            analytic = hilbert(imfs[i])
            IA[i] = np.abs(analytic)
            IP[i] = np.unwrap(np.angle(analytic))
            IF_normalized[i, 1:] = np.diff(IP[i]) / (2.0 * np.pi)
            IF_normalized[i, 0] = IF_normalized[i, 1]

        IF = IF_normalized * fps
        return IA, IF, IF_normalized, IP


def filter_if_physical(if_hz: np.ndarray, fps: float, med_kernel: int = 5) -> np.ndarray:
    """Median filter + clip to [0, Nyquist], same as legacy feature scripts."""
    x = np.asarray(if_hz, dtype=np.float32)
    k = med_kernel if med_kernel % 2 == 1 else med_kernel + 1
    if k >= 3 and x.size >= k:
        x = medfilt(x, kernel_size=k).astype(np.float32)
    return np.clip(x, 0.0, fps / 2.0).astype(np.float32)


def ia_if_feature_matrix(
    signal_1d: np.ndarray,
    *,
    fps: float = 30.0,
    n_imf: int = 5,
    emd_method: EMDMethod = "CEEMDAN",
    max_imf: int = 10,
    remove_mean: bool = True,
    if_med_kernel: int = 5,
    savgol_window: int | None = None,
    savgol_poly: int = 2,
    analyzer: HHTAnalyzer | None = None,
) -> np.ndarray:
    """
    Map a 1D clinically salient trajectory to (T, 2 * n_imf) with column order
    [IA_0, IF_0, IA_1, IF_1, ...], matching the dataloader HHT tensor layout.

    Args:
        signal_1d: length-T 1D sequence
        fps: sampling rate (Hz scale for IF)
        n_imf: number of leading IMFs to keep (pad zeros if fewer available)
    """
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    T = x.shape[0]
    if T < 4:
        return np.zeros((T, 2 * n_imf), dtype=np.float32)

    if remove_mean:
        x = x - np.mean(x)

    if savgol_window is not None:
        w = savgol_window
        if w >= 3 and w % 2 == 1 and T > w:
            x = savgol_filter(x, w, savgol_poly)

    hht = analyzer or HHTAnalyzer(emd_method=emd_method, max_imf=max_imf)
    imfs = hht.decompose_signal(x)
    IA, IF, _, _ = hht.compute_hilbert_spectrum(imfs, fps=fps)

    out = np.zeros((T, 2 * n_imf), dtype=np.float32)
    n_available = min(n_imf, IA.shape[0])
    for i in range(n_available):
        ia = IA[i].astype(np.float32)
        if_hz = filter_if_physical(IF[i], fps, med_kernel=if_med_kernel)
        out[:, 2 * i] = ia
        out[:, 2 * i + 1] = if_hz
    return out
