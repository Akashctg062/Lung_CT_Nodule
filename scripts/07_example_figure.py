"""Figure 1 helper - one benign and one malignant nodule under every simulated acquisition.
Examples are typical, not hand-picked: among nodules of near-median diameter for their class, the one whose
native-scan noise index is closest to the cohort median is shown.
Run: python scripts/07_example_figure.py --config configs/default.yaml"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml  # plain YAML read: this script needs no torch

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
with open(ap.parse_args().config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
df = pd.read_csv(os.path.join(cfg["work_dir"], "tables", "nodules.csv"))
bank = os.path.join(cfg["work_dir"], "bank")
df = df[[os.path.exists(os.path.join(bank, f"{u}.npz")) for u in df.uid]].copy()


def clean_sigma(uid):
    z = np.load(os.path.join(bank, f"{uid}.npz")); names = [str(n) for n in z["names"]]
    return float(z["sigma"][names.index("clean")]) if "clean" in names else np.nan


picks = []
for lab in (0, 1):
    d = df[df.label_hard == lab]
    lo, hi = d.diam_mm.quantile([0.4, 0.6])
    cand = d[(d.diam_mm >= lo) & (d.diam_mm <= hi)].copy()
    cand["s0"] = [clean_sigma(u) for u in cand.uid]
    target = np.nanmedian([clean_sigma(u) for u in d.uid.sample(min(len(d), 200), random_state=0)])
    picks.append(cand.iloc[int(np.nanargmin(np.abs(cand.s0.values - target)))])

show = ["clean", "d050_r0", "d025_r0", "d010_r0", "d005_r0", "ksmooth_d100", "ksharp_d100"]
TITLE = {"clean": "Native", "d050_r0": "50% dose", "d025_r0": "25% dose", "d010_r0": "10% dose",
         "d005_r0": "5% dose", "ksmooth_d100": "Smooth kernel", "ksharp_d100": "Sharp kernel"}
fig, ax = plt.subplots(2, len(show), figsize=(2.1 * len(show), 5.4))
for i, r in enumerate(picks):
    z = np.load(os.path.join(bank, f"{r.uid}.npz")); names = [str(n) for n in z["names"]]
    for j, c in enumerate(show):
        a = ax[i, j]; a.axis("off")
        if c not in names:
            continue
        k = names.index(c)
        a.imshow(np.clip(z["vol"][k][z["vol"].shape[1] // 2], -1000, 400), cmap="gray")
        a.set_title(f"{TITLE[c]}\nσ̂ = {z['sigma'][k]:.0f} HU", fontsize=8)
    ax[i, 0].text(-8, 24, [f"benign ({r.diam_mm:.0f} mm)", f"malignant ({r.diam_mm:.0f} mm)"][i],
                  rotation=90, va="center", fontsize=9)
    print(["benign", "malignant"][i], r.uid, f"diam {r.diam_mm:.1f} mm, native sigma {r.s0:.1f} HU")
fig.tight_layout()
p = os.path.join(cfg["work_dir"], "results", "figures", "fig1_acquisition_examples.png"); fig.savefig(p, dpi=800); print("saved", p)
