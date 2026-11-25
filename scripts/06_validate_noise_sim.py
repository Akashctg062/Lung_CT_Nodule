"""Step 6 (recommended) - validate / calibrate the low-dose simulator against REAL paired data.
Uses one (or more) chest case of the Mayo 'LDCT-and-Projection-data' collection (TCIA), which
provides full-dose images and images reconstructed from 10 %-dose projections of the SAME scan.
Because the 10 % data were produced by inserting noise into the same projections,
(low - full) is the real added noise. We compare its std and noise power spectrum with ours,
and return the effective I0 that matches them (put it into configs/default.yaml -> i0).
Run:  python scripts/06_validate_noise_sim.py --config configs/default.yaml [--full DIR --low DIR]
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dart.utils import load_config, wpath, get_device
from dart.dicom_io import load_series
from dart.ct_physics import ParallelBeamCT, insert_lowdose_noise, slice_i0, KERNEL_WINDOW
from dart.figs1 import make_figS1


def files_in(d):
    out = []
    for dp, _, fs in os.walk(d):
        out += [os.path.join(dp, f) for f in fs if not f.lower().endswith((".xml", ".csv", ".txt"))]
    return out


def radial_nps(noise, mask, roi=32):
    """mean 2-D NPS over ROIs mostly inside mask (relaxed if none qualify), radially averaged"""
    H, W = noise.shape[-2:]
    for need in (0.9, 0.6, 0.0):
        ps = []
        for img, m in zip(noise, mask):
            for r in range(0, H - roi + 1, roi // 2):
                for c in range(0, W - roi + 1, roi // 2):
                    if m[r:r + roi, c:c + roi].mean() >= need:
                        x = img[r:r + roi, c:c + roi]; x = x - x.mean()
                        ps.append(np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2 / roi ** 2)
        if len(ps) >= 8:
            break
    if not ps:
        return None
    P = np.mean(ps, 0)
    yy, xx = np.indices(P.shape); rr = np.hypot(yy - roi / 2, xx - roi / 2).astype(int)
    return (np.bincount(rr.ravel(), P.ravel()) / np.maximum(np.bincount(rr.ravel()), 1))[: roi // 2 + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--full"); ap.add_argument("--low")
    ap.add_argument("--dose", type=float, default=0.10); ap.add_argument("--n_slices", type=int, default=12)
    ap.add_argument("--per_case_figs", action="store_true", help="also write one figure per Mayo case")
    a = ap.parse_args(); cfg = load_config(a.config); dev = get_device(cfg)
    if a.full or not cfg.get("mayo_root"):
        cases = [("case", a.full or cfg["mayo_full_dir"], a.low or cfg["mayo_low_dir"])]
    else:
        root = cfg["mayo_root"]
        cases = [(c, os.path.join(root, c, "full_dose"), os.path.join(root, c, "low_dose")) for c in sorted(os.listdir(root))
                 if os.path.isdir(os.path.join(root, c, "full_dose")) and os.path.isdir(os.path.join(root, c, "low_dose"))]
    print("Mayo cases:", [c[0] for c in cases])
    allres = []
    for case, full_dir, low_dir in cases:
        r = run_case(a, cfg, dev, case, full_dir, low_dir); allres.append(r); print(json.dumps(r, indent=2))
    real = np.array([r["std_added_real_HU"] for r in allres]); sim = np.array([r["std_added_sim_HU"] for r in allres])
    r = sim / real
    mode = cfg.get("i0_mode", "constant")
    key = "i0_per_mAs" if mode == "per_mAs" else "i0"
    if all("nps_corr" in c for c in allres):
        print("NPS shape: peak real %.2f vs sim %.2f cycles/pixel (mean); correlation %.2f" % (
            np.mean([c["nps_peak_real"] for c in allres]), np.mean([c["nps_peak_sim"] for c in allres]),
            np.mean([c["nps_corr"] for c in allres])))
    summ = dict(cases=allres, i0_mode=mode, mean_ratio=float(np.mean(r)), sd_ratio=float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                min_ratio=float(r.min()), max_ratio=float(r.max()),
                config_key=key, config_value=float(cfg[key]), recommended_value=float(float(cfg[key]) * np.mean(r ** 2)),
                note=f"set {key} in configs/default.yaml to recommended_value and re-run; mean_ratio should then be ~1.0")
    json.dump(summ, open(wpath(cfg, "results", "noise_validation.json"), "w"), indent=2)
    print("\nSUMMARY (%s)  sim/real noise ratio = %.3f +/- %.3f (range %.2f-%.2f)  ->  recommended %s = %.0f"
          % (mode, summ["mean_ratio"], summ["sd_ratio"], summ["min_ratio"], summ["max_ratio"], key, summ["recommended_value"]))
    out, rep = make_figS1(os.path.join(cfg["work_dir"], "results"))
    print(f"composite Fig. S1 (representative case {rep}) -> {out}")


def run_case(a, cfg, dev, case, full_dir, low_dir):
    F_ = load_series(files_in(full_dir)); L_ = load_series(files_in(low_dir))
    # pair slices by z position
    zf, zl = F_["z_mm"], L_["z_mm"]
    mid = len(zf) // 2; pick = np.linspace(mid - len(zf) // 4, mid + len(zf) // 4, a.n_slices).astype(int)
    full = F_["vol_hu"][pick].astype(np.float32)
    low = np.stack([L_["vol_hu"][np.argmin(np.abs(zl - zf[i]))] for i in pick]).astype(np.float32)
    mask = np.stack([ndimage.binary_erosion((s > -50) & (s < 150), iterations=3) for s in full])
    real = low - full
    n = full.shape[-1]; ct = ParallelBeamCT(n, cfg["n_angles"], device=dev, angle_chunk=30)
    with torch.no_grad():
        i0 = torch.as_tensor(slice_i0(F_["mas"][pick], F_["meta"]["slice_thickness"], cfg), dtype=torch.float32)
        sim = insert_lowdose_noise(torch.from_numpy(full).to(dev), F_["spacing_xy"][1], a.dose, ct,
                                   i0=i0, sigma_e=float(cfg["sigma_e"]), window=KERNEL_WINDOW["standard"]).cpu().numpy() - full
    s_real, s_sim = real[mask].std(), sim[mask].std()
    i0_eff = float(cfg["i0"]) * (s_sim / s_real) ** 2  # only meaningful in i0_mode: constant
    n_real, n_sim = radial_nps(real, mask), radial_nps(sim, mask)
    res = dict(case=case, dose=a.dose, mean_mAs=float(np.nanmean(F_["mas"][pick])),
               std_added_real_HU=float(s_real), std_added_sim_HU=float(s_sim), ratio=float(s_sim / s_real))
    if n_real is not None and n_sim is not None:
        fr = np.arange(len(n_real)) / 32.0
        a_, b_ = n_real / n_real.sum(), n_sim / n_sim.sum()
        res.update(nps_peak_real=float(fr[2 + a_[2:].argmax()]), nps_peak_sim=float(fr[2 + b_[2:].argmax()]),
                   nps_corr=float(np.corrcoef(a_, b_)[0, 1]))
    k = len(pick) // 2
    cdir = wpath(cfg, "results", "noise_validation_cases"); os.makedirs(cdir, exist_ok=True)
    np.savez_compressed(os.path.join(cdir, f"{case}.npz"), img_real=real[k].astype(np.float32), img_sim=sim[k].astype(np.float32),
                        nps_real=n_real if n_real is not None else np.zeros(17), nps_sim=n_sim if n_sim is not None else np.zeros(17),
                        pixel_mm=float(F_["spacing_xy"][1]))
    if not a.per_case_figs:
        return res
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.clip(real[k], -150, 150), cmap="gray"); ax[0].set_title(f"Real added noise (Mayo {int(a.dose*100)}%)")
    ax[1].imshow(np.clip(sim[k], -150, 150), cmap="gray"); ax[1].set_title("Simulated added noise (ours)")
    for a_ in ax[:2]:
        a_.axis("off")
    if n_real is not None and n_sim is not None:
        f = np.arange(len(n_real)) / 32
        ax[2].plot(f, n_real / n_real.sum(), label="real"); ax[2].plot(f[:len(n_sim)], n_sim / n_sim.sum(), label="simulated")
        ax[2].set_xlabel("spatial frequency (cycles/pixel)"); ax[2].set_ylabel("normalised NPS"); ax[2].legend()
    fig.tight_layout()
    target = wpath(cfg, "results", "figures", f"figS1_noise_validation_{case}.png")
    try:
        fig.savefig(target, dpi=800)
    except OSError as e:  # e.g. the PNG is open in an image viewer / VS Code preview (Windows file lock)
        alt = target.replace(".png", f"_{int(time.time())}.png")
        print(f"  WARNING: could not overwrite {target} ({e}); saved as {alt} instead")
        fig.savefig(alt, dpi=800)
    plt.close(fig)
    return res


if __name__ == "__main__":
    main()
