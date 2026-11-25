"""Build the 'acquisition bank': for every nodule, 3-D isotropic crops under several
simulated acquisitions (dose levels x noise realisations x reconstruction-kernel proxies),
plus the blind noise estimate sigma_hat of each acquisition.

Noise is inserted on FULL axial slices (so the noise depends on the true patient attenuation
along every ray) and only afterwards cropped around the nodule.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .ct_physics import ParallelBeamCT, kernel_proxy, KERNEL_WINDOW, attenuation_factor, noise_from_ep, slice_i0
from .noise_estimator import estimate_noise_hu

_CT_CACHE: dict = {}


def get_ct(n: int, n_angles: int, device) -> ParallelBeamCT:
    key = (n, n_angles, str(device))
    if key not in _CT_CACHE:
        _CT_CACHE[key] = ParallelBeamCT(n, n_angles, device=device, angle_chunk=30)
    return _CT_CACHE[key]


def _seed(*parts) -> int:
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def crop_iso(slices_hu: np.ndarray, z_mm: np.ndarray, spacing_xy, cx: float, cy: float, cz: float,
             size: int = 48, iso_mm: float = 1.0) -> np.ndarray:
    """Trilinear resample of a (Z,H,W) stack (z positions z_mm, ascending) to an isotropic cube
    centred at (col=cx, row=cy, z=cz mm). Outside the stack -> -1000 HU."""
    off = (np.arange(size) - (size - 1) / 2.0) * iso_mm
    zt = cz + off
    zi = np.interp(zt, z_mm, np.arange(len(z_mm)), left=-1e3, right=-1e3)
    ri = cy + off / spacing_xy[0]
    ci = cx + off / spacing_xy[1]
    # cut the needed block first (much faster than resampling from the whole stack)
    r0 = max(int(np.floor(ri.min())) - 2, 0); r1 = min(int(np.ceil(ri.max())) + 3, slices_hu.shape[1])
    c0 = max(int(np.floor(ci.min())) - 2, 0); c1 = min(int(np.ceil(ci.max())) + 3, slices_hu.shape[2])
    ok = zi > -1e2
    z0 = max(int(np.floor(zi[ok].min())) - 1, 0) if ok.any() else 0
    z1 = min(int(np.ceil(zi[ok].max())) + 2, slices_hu.shape[0]) if ok.any() else 1
    block = slices_hu[z0:z1, r0:r1, c0:c1].astype(np.float32)
    Z, Rr, C = np.meshgrid(np.where(ok, zi - z0, -1e3), ri - r0, ci - c0, indexing="ij")
    return map_coordinates(block, [Z, Rr, C], order=1, mode="constant", cval=-1000.0)


@torch.no_grad()
def simulate_conditions(stack_hu: np.ndarray, pixel_mm: float, conds: list[dict], series_uid: str, i0_slices: np.ndarray,
                        n_angles: int, sigma_e: float, device, batch: int = 8) -> dict:
    """All acquisition conditions for a stack of full axial slices (Z,H,W) -> {cond name: float32 (Z,H,W)}.
    The forward projection (anatomy) is computed once per slice batch and shared by every condition;
    each condition has its own reproducible noise stream."""
    Z, H, W = stack_hu.shape
    n = max(H, W)
    ct = get_ct(n, n_angles, device)
    out = {c["name"]: np.empty((Z, H, W), dtype=np.float16) for c in conds}  # float16 keeps RAM ~1 GB
    gens = {}
    for c in conds:
        g = torch.Generator(device=device); g.manual_seed(_seed(series_uid, c["name"], int(c.get("rep", 0)))); gens[c["name"]] = g
    for i in range(0, Z, batch):
        x = torch.from_numpy(stack_hu[i:i + batch].astype(np.float32)).to(device)
        if (H, W) != (n, n):
            pad = torch.full((x.shape[0], n, n), -1000.0, device=device)
            pad[:, :H, :W] = x; x = pad
        noisy_needed = any(float(c["dose"]) < 0.999 for c in conds)
        ep = attenuation_factor(x, pixel_mm, ct) if noisy_needed else None
        i0 = torch.as_tensor(i0_slices[i:i + batch], dtype=torch.float32)
        for c in conds:
            dose, kernel = float(c["dose"]), c.get("kernel", "standard")
            y = x
            if dose < 0.999:
                y = x + noise_from_ep(ep, pixel_mm, dose, ct, i0, sigma_e, KERNEL_WINDOW.get(kernel, "shepp-logan"), gens[c["name"]])
            y = kernel_proxy(y, kernel)
            out[c["name"]][i:i + batch] = torch.clamp(y[:, :H, :W], -3000.0, 30000.0).float().cpu().numpy()
    return out


def build_series_bank(series: dict, nodules: list[dict], conditions: list[dict], cfg: dict, device) -> dict:
    """series: output of dicom_io.load_series; nodules: list of dict(uid,cx,cy,cz)
    returns {uid: dict(vol=int16 (C,S,S,S), sigma=float32 (C,), names=[...])}"""
    S = int(cfg.get("crop_size", 48)); half = S / 2.0 + 3.0
    z = series["z_mm"]; sp = series["spacing_xy"]; vol = series["vol_hu"]
    # union of needed slice indices
    need = np.zeros(len(z), bool)
    for nd in nodules:
        need |= (z >= nd["cz"] - half) & (z <= nd["cz"] + half)
    idx = np.where(need)[0]
    if idx.size == 0:
        return {}
    lo, hi = max(idx.min() - 1, 0), min(idx.max() + 2, len(z))
    stack = vol[lo:hi]; zz = z[lo:hi]
    res = {nd["uid"]: dict(vol=np.empty((len(conditions), S, S, S), np.int16),
                           sigma=np.empty(len(conditions), np.float32),
                           names=[c["name"] for c in conditions]) for nd in nodules}
    mas = series.get("mas", np.full(len(z), np.nan))[lo:hi]
    i0_slices = slice_i0(mas, series.get("meta", {}).get("slice_thickness", np.nan), cfg)
    sims = simulate_conditions(stack, sp[1], conditions, series.get("series_uid", "x"), i0_slices,
                               int(cfg.get("n_angles", 360)), float(cfg.get("sigma_e", 10.0)), device)
    for ci, cond in enumerate(conditions):
        sim = sims.pop(cond["name"])
        for nd in nodules:
            cube = crop_iso(sim, zz, sp, nd["cx"], nd["cy"], nd["cz"], S, float(cfg.get("iso_mm", 1.0)))
            res[nd["uid"]]["vol"][ci] = np.clip(np.round(cube), -1024, 3071).astype(np.int16)
            sel = np.where(np.abs(zz - nd["cz"]) <= 10.0)[0]
            if sel.size == 0:
                sel = np.array([int(np.argmin(np.abs(zz - nd["cz"])))])
            sel = sel[:: max(1, len(sel) // 6)][:6]
            res[nd["uid"]]["sigma"][ci] = estimate_noise_hu(sim[sel].astype(np.float32))
    return res
