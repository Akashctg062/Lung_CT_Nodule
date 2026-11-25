"""Graphical abstract (800 dpi JPG + TIFF). Laid out on a fixed 1200 x 560 grid (1 unit = 0.01 inch).
Run: python scripts/10_graphical_abstract.py --out work/results/figures --slices work/results/ga_nodule_slices.npy
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="work/results/figures"); ap.add_argument("--slices", default="work/results/ga_nodule_slices.npy")
a = ap.parse_args()

plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 8.5, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
INK, MUTED, EDGE, PANEL = "#1a1a19", "#5f5e5a", "#c9c8c2", "#f6f6f3"
DART, BASE, WN = "#2a78d6", "#eb6834", "#1baf7a"           # validated categorical slots 1-3
W, H = 1200, 560
fig = plt.figure(figsize=(W / 100, H / 100))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
ax_at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])


def panel(x, y, w, h, num, title, fill=PANEL, edge=EDGE, lw=0.9):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=fill, ec=edge, lw=lw))
    bg.add_patch(FancyBboxPatch((x + 12, y + h - 34), 22, 22, boxstyle="round,pad=0,rounding_size=11", fc=DART, ec="none"))
    bg.text(x + 23, y + h - 23, num, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    bg.text(x + 42, y + h - 23, title, ha="left", va="center", color=INK, fontsize=11, fontweight="bold")


def chip(x, y, w, h, head, lines, fc="white"):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=EDGE, lw=0.7))
    top = y + h / 2 + 26
    bg.text(x + 10, top, head, ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
    for i, t in enumerate(lines):
        bg.text(x + 10, top - 19 - 15 * i, t, ha="left", va="center", color=MUTED, fontsize=8)


def arrow(x0, x1, y):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=6,head_width=3.5", color="#8a8a85", lw=1.4,
                                 shrinkA=0, shrinkB=0, mutation_scale=1))


# ---------------- title
bg.text(20, 530, "Dose-consistency training keeps lung-nodule malignancy AI stable from native to 5% radiation dose",
        ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
bg.text(20, 504, "Open data only • physics-validated low-dose simulation • noise-conditional conformal prediction • "
        "5-fold patient-level CV + external test", ha="left", va="center", fontsize=9.5, color=MUTED)

TOP, BOT = 478, 92                         # panel band (shared top and bottom edges)
COLS = [(20, 228), (276, 246), (550, 246), (824, 356)]
for (x0, w0), (x1, _) in zip(COLS[:-1], COLS[1:]):
    arrow(x0 + w0 + 4, x1 - 4, (TOP + BOT) / 2)

# ---------------- 1 data
x, w = COLS[0]; panel(x, BOT, w, TOP - BOT, "1", "Open CT data")
cw, gap = w - 24, 16
chh = (TOP - BOT - 50 - 12 - 2 * gap) / 3
y0 = TOP - 50 - chh
chip(x + 12, y0, cw, chh, "LIDC-IDRI (development)", ["1,390 nodules, 695 patients", "≥3 radiologist ratings", "5-fold patient-level CV"])
chip(x + 12, y0 - chh - gap, cw, chh, "Mayo LDCT (validation)", ["6 paired full / 10%-dose", "chest cases (real noise)", "calibrates the simulator"])
chip(x + 12, y0 - 2 * (chh + gap), cw, chh, "LUNGx (external test)", ["83 nodules, other institution", "pathology / follow-up", "reference standard"])

# ---------------- 2 simulator
x, w = COLS[1]; panel(x, BOT, w, TOP - BOT, "2", "Physics dose simulator")
bg.text(x + 14, TOP - 56, "Noise inserted in projection space,", ha="left", va="center", fontsize=8.3, color=MUTED)
bg.text(x + 14, TOP - 71, "reconstructed by filtered back-projection", ha="left", va="center", fontsize=8.3, color=MUTED)
S = np.load(a.slices); labels = ["100%", "25%", "5%"]
iw, igap = 72, 8; ix0 = x + (w - (3 * iw + 2 * igap)) / 2; iy = TOP - 98 - iw
for i in range(3):
    ax = ax_at(ix0 + i * (iw + igap), iy, iw, iw)
    ax.imshow(np.clip(S[i], -1000, 400), cmap="gray", vmin=-1000, vmax=400, interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444444"); s.set_linewidth(0.6)
    bg.text(ix0 + i * (iw + igap) + iw / 2, iy + iw + 9, labels[i], ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
bg.text(x + w / 2, iy - 13, "same nodule, simulated dose (malignant, 16 mm)", ha="center", va="center", fontsize=7.5, color=MUTED)
ty = iy - 52
bg.add_patch(FancyBboxPatch((x + 12, ty - 92), w - 24, 92, boxstyle="round,pad=0,rounding_size=6", fc="white", ec=EDGE, lw=0.7))
bg.text(x + w / 2, ty - 24, "1.01 ± 0.06", ha="center", va="center", fontsize=17, fontweight="bold", color=DART)
bg.text(x + w / 2, ty - 50, "simulated / real noise ratio", ha="center", va="center", fontsize=8.3, color=INK)
bg.text(x + w / 2, ty - 66, "vs real 10%-dose scans (6 cases)", ha="center", va="center", fontsize=8, color=MUTED)
bg.text(x + w / 2, ty - 81, "50%, 25%, 10%, 5% dose + 4 kernel shifts", ha="center", va="center", fontsize=7.6, color=MUTED)

# ---------------- 3 DART training
x, w = COLS[2]; panel(x, BOT, w, TOP - BOT, "3", "DART training")
cx = x + w / 2
vy = TOP - 88
for i, (lab, sub) in enumerate([("View A", "native or low dose"), ("View B", "random low dose")]):
    bx = x + 16 + i * ((w - 32) / 2 + 4)
    bw = (w - 32) / 2 - 4
    bg.add_patch(FancyBboxPatch((bx, vy - 16), bw, 40, boxstyle="round,pad=0,rounding_size=6", fc="white", ec=EDGE, lw=0.7))
    bg.text(bx + bw / 2, vy + 11, lab, ha="center", va="center", fontsize=8.8, fontweight="bold", color=INK)
    bg.text(bx + bw / 2, vy - 5, sub, ha="center", va="center", fontsize=7.4, color=MUTED)
ny = vy - 78
bg.add_patch(FancyBboxPatch((x + 30, ny - 18), w - 60, 40, boxstyle="round,pad=0,rounding_size=6", fc=DART, ec="none", alpha=0.12))
bg.add_patch(FancyBboxPatch((x + 30, ny - 18), w - 60, 40, boxstyle="round,pad=0,rounding_size=6", fc="none", ec=DART, lw=1.2))
bg.text(cx, ny + 9, "Shared 3-D ResNet", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
bg.text(cx, ny - 7, "44 mm nodule cube, 1 mm voxels", ha="center", va="center", fontsize=7.6, color=MUTED)
for i in range(2):
    bx = x + 16 + i * ((w - 32) / 2 + 4) + ((w - 32) / 2 - 4) / 2
    bg.add_patch(FancyArrowPatch((bx, vy - 18), (cx + (-40 if i == 0 else 40), ny + 24), arrowstyle="-|>,head_length=5,head_width=3",
                                 color="#8a8a85", lw=1.1, mutation_scale=1))
ly = ny - 62
bg.add_patch(FancyArrowPatch((cx, ny - 20), (cx, ly + 22), arrowstyle="-|>,head_length=5,head_width=3", color="#8a8a85", lw=1.1, mutation_scale=1))
bg.add_patch(FancyBboxPatch((x + 22, ly - 18), w - 44, 40, boxstyle="round,pad=0,rounding_size=6", fc="white", ec=DART, lw=1.4))
bg.text(cx, ly + 9, "Dose-consistency loss", ha="center", va="center", fontsize=9, fontweight="bold", color=DART)
bg.text(cx, ly - 7, "predictions for A and B must agree", ha="center", va="center", fontsize=7.6, color=INK)
items = ["+ reader soft labels", "+ semantic-feature heads", r"Blind noise index $\hat{\sigma}$ picks the", "conformal threshold (Mondrian CP)"]
for i, t in enumerate(items):
    bg.text(x + 18, ly - 46 - 16 * i, t, ha="left", va="center", fontsize=8.1, color=INK if i < 2 else MUTED)

# ---------------- 4 results (2 x 2 mini panels on one grid)
x, w = COLS[3]; panel(x, BOT, w, TOP - BOT, "4", "Key results (1,192 test nodules)")
LM, GAPX, TT, GAPY, BM = 40, 52, 66, 66, 40         # left margin, column gap, top, row gap, bottom margin
pw = (w - LM - GAPX - 14) / 2; ph = (TOP - BOT - TT - GAPY - BM) / 2
c0, c1 = x + LM, x + LM + pw + GAPX; r1 = BOT + BM; r0 = r1 + ph + GAPY
pos = [(c0, r0), (c1, r0), (c0, r1), (c1, r1)]


def mini(i, title):
    px, py = pos[i]; ax = ax_at(px, py, pw, ph)
    bg.text(px - 30, py + ph + 15, title, ha="left", va="center", fontsize=8.3, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7, colors=MUTED, pad=2)
    ax.grid(axis="y", color="#e6e6e2", lw=0.6); ax.set_axisbelow(True)
    return ax


labs = ["Baseline", "White\nnoise", "DART"]; cols = [BASE, WN, DART]
ax = mini(0, "AUC vs simulated dose")
ax.plot(range(5), [0.909, 0.909, 0.908, 0.908, 0.906], "-o", color=DART, ms=3.2, lw=1.8)
ax.plot(range(5), [0.890, 0.889, 0.887, 0.879, 0.857], "-s", color=BASE, ms=3, lw=1.5)
ax.set_xticks(range(5)); ax.set_xticklabels(["100", "50", "25", "10", "5%"]); ax.set_xlim(-0.35, 4.35)
ax.set_ylim(0.84, 0.925); ax.set_yticks([0.85, 0.88, 0.91])
ax.text(0, 0.9165, "DART", color=DART, fontsize=7.2, fontweight="bold", ha="left", va="center")
ax.text(0, 0.8805, "Baseline", color=BASE, fontsize=7.2, fontweight="bold", ha="left", va="center")
ax.text(4, 0.9165, "0.906", color=DART, fontsize=7.2, fontweight="bold", ha="center", va="center")
ax.text(3.72, 0.857, "0.857", color=BASE, fontsize=7.2, fontweight="bold", ha="right", va="center")

ax = mini(1, "Decisions changed at 5% (%)")
vals = [15.9, 6.8, 2.8]; ax.bar(range(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.8); ax.set_ylim(0, 19); ax.set_yticks([0, 5, 10, 15])

ax = mini(2, "Calls kept down to 5% (%)")
vals = [82.0, 92.7, 96.1]; ax.bar(range(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.8); ax.set_ylim(70, 101); ax.set_yticks([70, 80, 90, 100])

ax = mini(3, "Coverage in noisiest 10%")
vals = [0.916, 0.749, 0.714]; cc = ["#9ec5f4", DART, "#184f95"]
ax.bar(range(3), vals, color=cc, width=0.62)
ax.axhline(0.90, color=INK, ls="--", lw=0.9, zorder=3)
ax.text(2.4, 0.925, "target 0.90", fontsize=6.6, color=INK, ha="right", va="center")
for i, v in enumerate(vals):
    ax.text(i, 0.6 + (v - 0.6) / 2, f"{v:.2f}", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK if i == 0 else "white")
ax.set_xticks(range(3)); ax.set_xticklabels(["Benign", "Malignant", "Malignant\n(real scans)"], fontsize=6.6)
ax.set_ylim(0.6, 0.95); ax.set_yticks([0.6, 0.7, 0.8, 0.9])

# ---------------- take-home band (aligned with the panel band)
bx0, bx1 = COLS[0][0], COLS[3][0] + COLS[3][1]
bg.add_patch(FancyBboxPatch((bx0, 18), bx1 - bx0, 58, boxstyle="round,pad=0,rounding_size=10", fc=INK, ec="none"))
bg.text(bx0 + 18, 56, "Take-home", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#9ec5f4")
bg.text(bx0 + 108, 56, "Agreement across doses, not realistic noise, makes nodule AI dose-robust (AUC 0.909 → 0.906; "
        "flips 2.8% vs 15.9%).", ha="left", va="center", fontsize=9.2, color="white")
bg.text(bx0 + 108, 36, "Conformal guarantees hold on average but fail for cancers in the noisiest scans and at a new site "
        "→ class-, noise- and site-specific calibration.", ha="left", va="center", fontsize=9.2, color="white")

os.makedirs(a.out, exist_ok=True)
png = os.path.join(a.out, "graphical_abstract.png")
fig.savefig(png, dpi=800, facecolor="white"); plt.close(fig)
im = Image.open(png).convert("RGB")
im.save(os.path.join(a.out, "graphical_abstract.jpg"), quality=95, dpi=(800, 800), subsampling=0)
im.save(os.path.join(a.out, "graphical_abstract.tif"), compression="tiff_lzw", dpi=(800, 800))
print("saved", im.size, "->", a.out)
