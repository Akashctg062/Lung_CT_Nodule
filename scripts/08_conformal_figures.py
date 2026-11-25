"""Step 8 - publication versions (800 dpi) of Fig. 2, 3, 4, 5 and 6 (reads the CSVs written by 04_evaluate.py; run it AFTER 04).
fig3_conformal_alpha<a>.png : coverage and 'uncertain' rate vs simulated dose (honest 0.80-0.95 axis)
fig6_coverage_by_noise_alpha<a>.png : coverage vs noise stratum (where standard CP fails), DART + white-noise
Run: python scripts/08_conformal_figures.py --config configs/default.yaml
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml  # plain YAML read: this script needs no torch

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
with open(ap.parse_args().config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
res = os.path.join(cfg["work_dir"], "results"); fig_dir = os.path.join(res, "figures"); os.makedirs(fig_dir, exist_ok=True)
DOSE = {"clean": 100, "d050_r0": 50, "d025_r0": 25, "d010_r0": 10, "d005_r0": 5}
NAME = {"standard": "Standard CP (native calibration)", "pooled": "Pooled CP (dose-augmented)",
        "mondrian": "Noise-Mondrian CP, 4 bins", "mondrian_cc": "Noise x class Mondrian CP"}
STYLE = {"standard": ("#1f77b4", "o", "-"), "pooled": ("#7f7f7f", "s", "--"),
         "mondrian": ("#d62728", "D", "-"), "mondrian_cc": ("#ff7f0e", "^", ":")}

NICE = {"baseline": "Baseline", "gauss": "+White-noise aug.", "physics": "+Physics aug.",
        "physics_cons": "+Dose consistency", "physics_cons_soft": "+Reader soft labels", "dart": "DART (full)"}
R = pd.read_csv(os.path.join(res, "discrimination.csv"))
fig, ax = plt.subplots(figsize=(6, 4.2))
for v in [x for x in NICE if x in R.variant.unique()]:
    r = R[R.variant == v].set_index("condition").reindex(list(DOSE)).dropna(subset=["AUC"])
    xs = [DOSE[c] for c in r.index]
    ln, = ax.plot(xs, r.AUC, marker="o", label=NICE[v], lw=2.4 if v == "dart" else 1.4)
    if v in ("dart", "baseline"):  # 95% CI band in the line's own colour
        ax.fill_between(xs, r.AUC_lo, r.AUC_hi, alpha=0.15, color=ln.get_color(), lw=0)
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks(list(DOSE.values())); ax.set_xticklabels([f"{v}%" for v in DOSE.values()])
ax.minorticks_off(); ax.set_xlabel("Simulated dose (% of native acquisition)"); ax.set_ylabel("AUC (test folds pooled)")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left"); fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig2_auc_vs_dose.png"), dpi=800); plt.close(fig)

C = pd.read_csv(os.path.join(res, "conformal.csv"))
for alpha in sorted(C.alpha.unique()):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for m in ["standard", "pooled", "mondrian", "mondrian_cc"]:
        s = C[(C.alpha == alpha) & (C.variant == "dart") & (C.method == m) & C.condition.isin(DOSE)].copy()
        if s.empty:
            continue
        s["x"] = s.condition.map(DOSE); s = s.sort_values("x", ascending=False)
        col, mk, ls = STYLE[m]
        axs[0].plot(s.x, s.coverage, marker=mk, ls=ls, color=col, label=NAME[m])
        axs[1].plot(s.x, s.uncertain_rate, marker=mk, ls=ls, color=col, label=NAME[m])
    axs[0].axhline(1 - alpha, ls="--", c="k", lw=1); axs[0].text(100, 1 - alpha + 0.004, f"target {1 - alpha:.2f}", fontsize=8)
    axs[0].set_ylim(1 - alpha - 0.10, min(1.0, 1 - alpha + 0.05)); axs[0].set_ylabel("Empirical coverage")
    axs[1].set_ylabel("'Uncertain: refer' rate")
    for a in axs:
        a.set_xscale("log"); a.invert_xaxis(); a.set_xticks(list(DOSE.values())); a.set_xticklabels([f"{v}%" for v in DOSE.values()])
        a.minorticks_off(); a.set_xlabel("Simulated dose (% of native)"); a.grid(alpha=.3)
    axs[0].legend(fontsize=7, loc="lower left"); axs[0].set_title("A  Coverage (DART)", loc="left", fontsize=10)
    axs[1].set_title("B  Scans flagged uncertain (DART)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"fig3_conformal_alpha{alpha}.png"), dpi=800); plt.close(fig)

S = pd.read_csv(os.path.join(res, "conformal_by_noise_stratum.csv"))
order = ["Q1 lowest", "Q2", "Q3", "Q4 (75-90%)", "top 10%"]
lab = ["Q1\n(least noise)", "Q2", "Q3", "Q4", "top 10%\n(most noise)"]
for alpha in sorted(S.alpha.unique()):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, v, title in [(axs[0], "dart", "A  DART"), (axs[1], "gauss", "B  White-noise augmentation")]:
        for m, col, mk, nm in [("standard", "#1f77b4", "o", "Standard CP"), ("mondrian_K4", "#d62728", "D", "Noise-Mondrian, 4 bins"),
                               ("mondrian_K8", "#9467bd", "v", "Noise-Mondrian, 8 bins (post hoc)")]:
            s = S[(S.alpha == alpha) & (S.variant == v) & (S.method == m)].set_index("stratum").reindex(order)
            if s.coverage.isna().all():
                continue
            a.plot(range(5), s.coverage, marker=mk, color=col, label=nm)
        a.axhline(1 - alpha, ls="--", c="k", lw=1)
        a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=8); a.set_xlabel("Noise stratum of the acquisition (blind noise index)")
        a.set_title(title, loc="left", fontsize=10); a.grid(alpha=.3)
    axs[0].set_ylabel("Empirical coverage"); axs[0].set_ylim(1 - alpha - 0.15, min(1.0, 1 - alpha + 0.05))
    axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"fig6_coverage_by_noise_alpha{alpha}.png"), dpi=800); plt.close(fig)
# ---- Fig. 4 and Fig. 5 from the saved test predictions (no torch needed)
import glob
import numpy as np
LAB = {"clean": "Native", "d050_r0": "50%", "d025_r0": "25%", "d010_r0": "10%", "d005_r0": "5%",
       "ksmooth_d100": "Smooth\n100%", "ksharp_d100": "Sharp\n100%", "ksmooth_d025": "Smooth\n25%", "ksharp_d025": "Sharp\n25%"}
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(cfg["work_dir"], "preds", "*.csv")))], ignore_index=True)
T = P[(P.split == "test") & (P.label_hard >= 0)]
d = T[T.variant == "dart"]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.boxplot([d[d.condition == c].sigma.dropna() for c in LAB], showfliers=False, widths=0.55,
           medianprops=dict(color="#2a78d6", lw=1.6), boxprops=dict(color="#333333"), whiskerprops=dict(color="#333333"),
           capprops=dict(color="#333333"))
ax.set_xticks(range(1, len(LAB) + 1)); ax.set_xticklabels(list(LAB.values()), fontsize=8)
ax.axvline(5.5, color="#999999", lw=0.8, ls=":")
ax.text(3, ax.get_ylim()[1] * 0.97, "Simulated dose", ha="center", va="top", fontsize=8, color="#555555")
ax.text(7.5, ax.get_ylim()[1] * 0.97, "Kernel shifts (unseen)", ha="center", va="top", fontsize=8, color="#555555")
ax.set_ylabel(r"Blind noise index $\hat{\sigma}$ (HU)"); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig4_sigma_by_condition.png"), dpi=800); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4.4))
for v, col in [("baseline", "#eb6834"), ("dart", "#2a78d6")]:
    for c, ls, mk, nm in [("clean", "-", "o", "native dose"), ("d005_r0", "--", "s", "5% dose")]:
        dd = T[(T.variant == v) & (T.condition == c)]
        b = np.clip((dd.prob * 10).astype(int), 0, 9)
        ax.plot(dd.groupby(b).prob.mean(), dd.groupby(b).label_hard.mean(), marker=mk, ms=4, lw=1.5, ls=ls, color=col,
                label=f"{NICE[v]}, {nm}")
ax.plot([0, 1], [0, 1], color="#555555", ls=":", lw=1); ax.set_xlabel("Predicted probability of malignancy")
ax.set_ylabel("Observed fraction malignant"); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig5_reliability.png"), dpi=800); plt.close(fig)
print("figures written to", fig_dir)
