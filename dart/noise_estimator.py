"""Blind (reference-free) CT noise estimator used for noise-conditional conformal prediction.

sigma_hat = robust standard deviation of a high-pass residual, measured only inside
homogeneous soft tissue (smoothed HU in [-50, 150], eroded) of the full axial slices around the nodule.
The residual r = x - mean(4-neighbours); for white noise Var(r) = 1.25 sigma^2.
The estimate does NOT need the dose, the scanner, or a reference image, so it can be computed
for any new scan at inference time.
"""
import numpy as np
from scipy import ndimage


def estimate_noise_hu(slices_hu: np.ndarray, lo: float = -50.0, hi: float = 150.0,
                      erode_iter: int = 3, max_voxels: int = 400_000, rng: int = 0) -> float:
    x = np.asarray(slices_hu, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    vals = []
    for s in x:
        # select soft tissue on a SMOOTHED copy: thresholding the noisy image itself would discard the
        # noisiest pixels (biasing sigma low) and, at very low dose, leave no mask at all
        sm = ndimage.uniform_filter(s, size=7)
        mask = (sm > lo) & (sm < hi)
        mask = ndimage.binary_erosion(mask, iterations=erode_iter)
        if mask.sum() < 200:
            continue
        nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)) / 4.0
        r = (s - nb)[mask]
        vals.append(r)
    if not vals:
        return float("nan")
    r = np.concatenate(vals)
    if r.size > max_voxels:
        r = np.random.default_rng(rng).choice(r, max_voxels, replace=False)
    mad = np.median(np.abs(r - np.median(r)))
    return float(1.4826 * mad / np.sqrt(1.25))
