"""Composite supplementary Fig. S1 (simulator validation on the Mayo 10%-dose cases).
Reads results/noise_validation.json and results/noise_validation_cases/<case>.npz (written by step 06).
numpy + matplotlib only (no torch), so it can be re-drawn quickly with scripts/06b_figure_S1.py.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

REAL, SIM = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2


def make_figS1(results_dir, dpi=800):
    summ = json.load(open(os.path.join(results_dir, "noise_validation.json")))
    cases = [c["case"] for c in summ["cases"]]
    D = {c: np.load(os.path.join(results_dir, "noise_validation_cases", f"{c}.npz")) for c in cases}
    ratio = np.array([c["ratio"] for c in summ["cases"]])
    rep = cases[int(np.argmin(np.abs(ratio - ratio.mean())))]          # representative case: ratio closest to the mean
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

    # A, B: added-noise images of the representative case (central crop, identical window)
    d = D[rep]; px = float(d["pixel_mm"])
    for j, (key, ttl) in enumerate([("img_real", "A  Real added noise (Mayo 10% dose)"), ("img_sim", "B  Simulated added noise (ours)")]):
        ax = fig.add_subplot(gs[0, j]); im = d[key]; H, W = im.shape; c = min(H, W, 256)
        crop = im[(H - c) // 2:(H + c) // 2, (W - c) // 2:(W + c) // 2]
        ax.imshow(np.clip(crop, -150, 150), cmap="gray", vmin=-150, vmax=150, interpolation="nearest"); ax.axis("off")
        bar = 20.0 / px                                                     # 2 cm scale bar
        halo = [pe.withStroke(linewidth=2.2, foreground="black")]
        ax.plot([c - 12 - bar, c - 12], [c - 12, c - 12], color="white", lw=2.5, solid_capstyle="butt",
                path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
        ax.text(c - 12 - bar / 2, c - 26, "2 cm", color="white", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", path_effects=halo)
        ax.set_title(ttl, loc="left", fontsize=9, fontweight="bold")
    fig.text(0.5, 0.515, f"Case {rep}, window −150 to 150 HU", ha="center", fontsize=7, color="#555555")

    # C: noise magnitude, simulated vs real, all cases
    ax = fig.add_subplot(gs[1, 0])
    r = np.array([c["std_added_real_HU"] for c in summ["cases"]]); s = np.array([c["std_added_sim_HU"] for c in summ["cases"]])
    lo, hi = min(r.min(), s.min()) * 0.9, max(r.max(), s.max()) * 1.08
    ax.plot([lo, hi], [lo, hi], ls="--", color="#555555", lw=0.9, label="identity")
    ax.scatter(r, s, s=26, color=REAL, zorder=3)
    span = hi - lo; off = {}
    for i in range(len(cases)):                      # nudge labels of neighbouring points apart
        near = [j for j in range(len(cases)) if j != i and abs(r[j] - r[i]) < 0.08 * span and abs(s[j] - s[i]) < 0.08 * span]
        off[i] = (5, 4) if near and all(s[i] >= s[j] for j in near) else ((5, -9) if near else (5, -3))
    for i, (cs, x, y) in enumerate(zip(cases, r, s)):
        ax.annotate(cs, (x, y), xytext=off[i], textcoords="offset points", fontsize=6.5, color="#333333")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Real added noise SD (HU)"); ax.set_ylabel("Simulated added noise SD (HU)")
    ax.text(0.04, 0.96, f"sim/real = {summ['mean_ratio']:.2f} ± {summ['sd_ratio']:.2f}\n"
                        f"(range {summ['min_ratio']:.2f}–{summ['max_ratio']:.2f}; {len(cases)} cases)",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.grid(alpha=0.3); ax.set_title("C  Noise magnitude", loc="left", fontsize=9, fontweight="bold")

    # D: normalised noise power spectra, mean over cases with min-max band
    ax = fig.add_subplot(gs[1, 1])
    for key, col, nm in [("nps_real", REAL, "Real, Mayo 10% dose"), ("nps_sim", SIM, "Simulated")]:
        Y = np.stack([D[c][key] / D[c][key].sum() for c in cases]); f = np.arange(Y.shape[1]) / 32.0
        ax.fill_between(f, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0)
        ax.plot(f, Y.mean(0), color=col, lw=1.8, label=f"{nm} (mean, range)")
    corr = np.mean([c["nps_corr"] for c in summ["cases"]])
    ax.text(0.03, 0.96, f"spectral correlation r = {corr:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("Spatial frequency (cycles/pixel)"); ax.set_ylabel("Normalised NPS")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.grid(alpha=0.3)
    ax.set_title("D  Noise texture (power spectrum)", loc="left", fontsize=9, fontweight="bold")

    out = os.path.join(results_dir, "figures", "figS1_noise_validation.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out, rep
