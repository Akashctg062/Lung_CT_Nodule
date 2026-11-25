"""Step 9 - three further analyses on the saved test predictions (no GPU, no torch, no scipy).

  A. Per-nodule dose tolerance: the lowest simulated dose down to which each nodule's call survives,
     by Lung-RADS size band, and what predicts a flip (clustered logistic regression).
  B. Repairing the conformal guarantee: cluster-weighted noise-Mondrian CP and noise-localised CP,
     plus a real-noise test on native scans only.
  C. Semantic-feature dose fragility: DART's six semantic heads against reader ratings at every dose.

Writes CSVs to work/results/extra/ and figures fig7-fig9 (800 dpi) to work/results/figures/.
Run: python scripts/09_extra_analyses.py --config configs/default.yaml
"""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dart.conformal import NoiseMondrianCP, StandardCP, lac_scores, predict_sets

DPI = 800
SEED = 2026
N_BOOT = 1000
DOSES = ["clean", "d050_r0", "d025_r0", "d010_r0", "d005_r0"]
DOSE_PCT = {"clean": 100, "d050_r0": 50, "d025_r0": 25, "d010_r0": 10, "d005_r0": 5}
COND_ALL = DOSES + ["ksmooth_d100", "ksharp_d100", "ksmooth_d025", "ksharp_d025"]
NICE = {"baseline": "Baseline", "gauss": "White-noise aug.", "physics": "Physics aug.",
        "physics_cons": "Dose consistency", "physics_cons_soft": "Reader soft labels", "dart": "DART"}
VCOL = {"dart": "#2a78d6", "baseline": "#eb6834", "gauss": "#1baf7a"}      # validated categorical slots 1-3
VMK = {"dart": "o", "baseline": "s", "gauss": "^"}
BANDS = [(0, 6, "<6 mm"), (6, 8, "6-<8 mm"), (8, 15, "8-<15 mm"), (15, 999, "≥15 mm")]  # Lung-RADS solid size cut-offs
BAND_COL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]                     # ordinal blue ramp (validated --ordinal)
SEM = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
SEM_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEM_MK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "grid.color": "#dddddd", "grid.linewidth": 0.6})

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
with open(ap.parse_args().config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
W = cfg["work_dir"]; OUT = os.path.join(W, "results", "extra"); FIG = os.path.join(W, "results", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(SEED)

P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(W, "preds", "*.csv")))], ignore_index=True)
P = P[P.label_hard >= 0]
NOD = pd.read_csv(os.path.join(W, "tables", "nodules.csv")).set_index("uid")
T = P[P.split == "test"]
VARIANTS = [v for v in NICE if v in P.variant.unique()]


# ------------------------------------------------------------------ helpers
def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def cluster_boot(df, fn, cluster="patient_id", n=N_BOOT):
    """percentile CI of fn(df) over patient-level bootstrap resamples"""
    g = df.groupby(cluster).indices; keys = np.array(list(g)); out = []
    for _ in range(n):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        out.append(fn(df.iloc[idx]))
    return np.nanpercentile(out, [2.5, 97.5])


def logit_cluster(X, y, groups, names):
    """logistic regression by IRLS with cluster-robust (sandwich) standard errors"""
    X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b)); Wd = p * (1 - p)
        H = X.T @ (X * Wd[:, None]); step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ b)); Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    sc = X * (y - p)[:, None]; meat = np.zeros_like(Hinv)
    for gidx in pd.Series(np.arange(len(y))).groupby(groups).indices.values():
        s = sc[gidx].sum(0); meat += np.outer(s, s)
    G = len(set(groups)); V = Hinv @ meat @ Hinv * G / (G - 1); se = np.sqrt(np.diag(V))
    rows = []
    for j, nm in enumerate(["intercept"] + names):
        rows.append(dict(term=nm, OR=math.exp(b[j]), lo=math.exp(b[j] - 1.96 * se[j]),
                         hi=math.exp(b[j] + 1.96 * se[j]), p=norm_p(b[j] / se[j])))
    return pd.DataFrame(rows)


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def wquantile_sorted(s_sorted, w_sorted, level):
    """smallest s with cumulative normalised weight >= level (weights already include the +inf test mass)"""
    c = np.cumsum(w_sorted, axis=-1)
    k = (c < level[..., None]).sum(-1)
    return np.where(k >= s_sorted.shape[-1], 1.0, s_sorted[np.minimum(k, s_sorted.shape[-1] - 1)])


# ================================================================== A. dose tolerance
wide = T[T.condition.isin(DOSES)].pivot_table(index=["variant", "uid", "patient_id", "fold"], columns="condition",
                                               values="prob").reset_index()
sig0 = T[T.condition == "clean"].groupby("uid").sigma.first()
wide["diam"] = wide.uid.map(NOD.diam_mm); wide["band"] = pd.cut(wide.diam, [b[0] for b in BANDS] + [999], right=False,
                                                                  labels=[b[2] for b in BANDS])
for k in ["texture", "spiculation", "margin"]:
    wide[k] = wide.uid.map(NOD[f"sem_{k}"])
wide["sigma0"] = wide.uid.map(sig0); wide["label"] = wide.uid.map(NOD.label_hard)
wide["margin_p"] = (wide["clean"] - 0.5).abs()
dec0 = wide["clean"] >= 0.5
stable = {c: ((wide[c] >= 0.5) == dec0) for c in DOSES[1:]}
# tolerance = lowest dose d such that the call is unchanged at every simulated dose >= d (100 = needs native dose)
tol = pd.Series(100, index=wide.index)
ok = pd.Series(True, index=wide.index)
for c in DOSES[1:]:
    ok &= stable[c]; tol = tol.where(~ok, DOSE_PCT[c])
wide["tolerance"] = tol; wide["flip_by5"] = (tol > 5).astype(int)
wide.to_csv(os.path.join(OUT, "dose_tolerance_per_nodule.csv"), index=False)

rows = []
for v in VARIANTS:
    for bnd in [None] + [b[2] for b in BANDS]:
        d = wide[(wide.variant == v) & ((wide.band == bnd) if bnd else True)]
        r = dict(variant=v, size_band=bnd or "all", n=len(d))
        for c in DOSES[1:]:
            r[f"survive_to_{DOSE_PCT[c]}"] = float((d.tolerance <= DOSE_PCT[c]).mean())
        lo, hi = cluster_boot(d, lambda x: (x.tolerance <= 5).mean(), n=500)
        r["survive_to_5_lo"], r["survive_to_5_hi"] = lo, hi
        rows.append(r)
TOLS = pd.DataFrame(rows); TOLS.to_csv(os.path.join(OUT, "dose_tolerance_by_size.csv"), index=False)

# paired difference DART - comparator in the share surviving to 5%, overall and per band
prs = []
for ref in ["baseline", "gauss"]:
    for bnd in [None] + [b[2] for b in BANDS]:
        a = wide[(wide.variant == "dart") & ((wide.band == bnd) if bnd else True)].set_index("uid")
        b = wide[(wide.variant == ref)].set_index("uid").loc[a.index]
        dd = pd.DataFrame({"patient_id": a.patient_id, "x": (a.tolerance <= 5).astype(float) - (b.tolerance <= 5).astype(float)})
        bs = []
        g = dd.groupby("patient_id").indices; keys = np.array(list(g))
        for _ in range(N_BOOT):
            idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)]); bs.append(dd.x.values[idx].mean())
        bs = np.array(bs)
        prs.append(dict(comparison=f"DART - {NICE[ref]}", size_band=bnd or "all", n=len(dd), diff=dd.x.mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
pd.DataFrame(prs).to_csv(os.path.join(OUT, "dose_tolerance_paired.csv"), index=False)

# what predicts losing the call by 5% dose?
terms = ["native decision margin (per 0.1)", "diameter (per doubling)", "sub-solid texture (reader mean <4)",
         "spiculation (per reader point)", "margin sharpness (per reader point)", "native noise σ̂ (per doubling)"]
lr = []
for v in ["dart", "baseline", "gauss"]:
    d = wide[wide.variant == v].dropna(subset=["diam", "texture", "spiculation", "margin", "sigma0"])
    X = np.column_stack([d.margin_p / 0.1, np.log2(d.diam), (d.texture < 4).astype(float), d.spiculation, d.margin,
                         np.log2(d.sigma0.clip(lower=1))])
    m = logit_cluster(X, d.flip_by5.values.astype(float), d.patient_id.values, terms); m.insert(0, "variant", v)
    m["n"] = len(d); m["events"] = int(d.flip_by5.sum()); lr.append(m)
LR = pd.concat(lr); LR.to_csv(os.path.join(OUT, "dose_tolerance_logistic.csv"), index=False)

# a native-dose confidence rule: smallest margin m such that >=99% of nodules with margin >= m keep their call to 5%
rule = []
for v in VARIANTS:
    d = wide[wide.variant == v].sort_values("margin_p")
    for m in np.round(np.arange(0.0, 0.50, 0.01), 2):
        s = d[d.margin_p >= m]
        if len(s) and (s.tolerance <= 5).mean() >= 0.99:
            rule.append(dict(variant=v, margin_threshold=m, share_of_nodules=len(s) / len(d),
                             keep_rate=(s.tolerance <= 5).mean(), n=len(s))); break
    else:
        rule.append(dict(variant=v, margin_threshold=np.nan, share_of_nodules=0.0, keep_rate=np.nan, n=0))
pd.DataFrame(rule).to_csv(os.path.join(OUT, "dose_tolerance_margin_rule.csv"), index=False)
# cross-validated version: threshold chosen on four folds, applied to the held-out fold
cvr = []
for v in VARIANTS:
    held = []
    for f in sorted(wide.fold.unique()):
        tr = wide[(wide.variant == v) & (wide.fold != f)]; te_ = wide[(wide.variant == v) & (wide.fold == f)]; thr = 0.5
        for m in np.round(np.arange(0.0, 0.50, 0.01), 2):
            s_ = tr[tr.margin_p >= m]
            if len(s_) and (s_.tolerance <= 5).mean() >= 0.99:
                thr = m; break
        sel = te_[te_.margin_p >= thr]; held.append(dict(thr=thr, n_sel=len(sel), n=len(te_), kept=int((sel.tolerance <= 5).sum())))
    H = pd.DataFrame(held)
    cvr.append(dict(variant=v, thresholds=";".join(f"{t:.2f}" for t in H.thr), share_selected=H.n_sel.sum() / H.n.sum(),
                    keep_rate_selected=H.kept.sum() / max(H.n_sel.sum(), 1), n_selected=int(H.n_sel.sum())))
pd.DataFrame(cvr).to_csv(os.path.join(OUT, "dose_tolerance_margin_rule_cv.csv"), index=False)

# ---- Fig. 7
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.42)
xs = [100, 50, 25, 10, 5]
for j, v in enumerate(["dart", "baseline", "gauss"]):
    ax = fig.add_subplot(gs[0, j])
    for (lo_, hi_, nm), col, mk in zip(BANDS, BAND_COL, ["o", "s", "^", "D"]):
        r = TOLS[(TOLS.variant == v) & (TOLS.size_band == nm)].iloc[0]
        ys = [1.0] + [r[f"survive_to_{x}"] for x in xs[1:]]
        ax.plot(xs, ys, marker=mk, ms=3.5, lw=1.4, color=col, label=f"{nm} (n={int(r.n)})")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs]); ax.minorticks_off()
    ax.set_ylim(0.70, 1.005); ax.grid(alpha=1); ax.set_xlabel("Simulated dose")
    ax.set_title(f"{'ABC'[j]}  {NICE[v]}", loc="left", fontsize=9, fontweight="bold")
    if j == 0:
        ax.set_ylabel("Nodules whose call survives\ndown to this dose")
        ax.legend(fontsize=6, loc="lower left", frameon=False, title="Size band", title_fontsize=6)
# D: probability of keeping the call to 5% vs native decision margin
ax = fig.add_subplot(gs[1, 0:2])
edges = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.5001]); mids = (edges[:-1] + edges[1:]) / 2
for v in ["dart", "gauss", "baseline"]:
    d = wide[wide.variant == v]; b = np.digitize(d.margin_p, edges) - 1
    keep = [((d.tolerance <= 5)[b == k]).mean() for k in range(len(mids))]
    ax.plot(mids, keep, marker=VMK[v], ms=4, lw=1.6, color=VCOL[v], label=NICE[v])
ax.axhline(0.99, ls="--", lw=0.8, color="#555555"); ax.text(0.005, 0.992, "99%", fontsize=6.5, color="#555555", va="bottom")
ax.set_xlabel("Decision margin at native dose, |p − 0.5|"); ax.set_ylabel("Call kept down to 5% dose")
ax.set_ylim(0.0, 1.03); ax.grid(alpha=1); ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_title("D  Confidence at native dose predicts dose tolerance", loc="left", fontsize=9, fontweight="bold")
# E: odds ratios (DART vs baseline)
ax = fig.add_subplot(gs[1, 2])
short = ["Margin +0.1", "Diameter ×2", "Sub-solid", "Spiculation +1", "Sharp margin +1", r"Native $\hat{\sigma}$ ×2"]
for k, (v, off) in enumerate([("dart", -0.15), ("baseline", 0.15)]):
    m = LR[(LR.variant == v) & (LR.term != "intercept")].reset_index(drop=True)
    yy = np.arange(len(m)) + off
    ax.errorbar(m.OR, yy, xerr=[m.OR - m.lo, m.hi - m.OR], fmt=VMK[v], ms=3.5, color=VCOL[v], lw=1, capsize=1.5, label=NICE[v])
ax.axvline(1, color="#555555", lw=0.8); ax.set_xscale("log"); ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=6.5)
ax.invert_yaxis(); ax.set_xlabel("Odds ratio of losing the call\nby 5% dose (95% CI)"); ax.grid(alpha=1, axis="x")
ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2); ax.set_title("E  Predictors of a flip", loc="left", fontsize=9, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig7_dose_tolerance.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)
print("A. dose tolerance done")


# ================================================================== B. conformal repairs
CAL_CONDS = DOSES
qe = np.nanquantile(T.drop_duplicates(["uid", "condition"]).sigma.values, [.25, .5, .75, .9])
STRATA = ["Q1 lowest", "Q2", "Q3", "Q4 (75-90%)", "top 10%"]


def fin(s):
    s = np.asarray(s, float); return np.where(np.isfinite(s), s, np.inf)


class ClusterMondrianCP(NoiseMondrianCP):
    """noise-Mondrian CP whose bin threshold weights each NODULE equally (weight 1/m_i for its m_i acquisitions in
    the bin) and uses the number of distinct nodules as n (CDF pooling, Dunn, Wasserman & Ramdas 2023)"""
    def fit_c(self, p, y, sigma, cluster):
        sigma = fin(sigma); f = sigma[np.isfinite(sigma)]
        self.edges = np.quantile(f, np.linspace(0, 1, self.n_bins + 1)[1:-1]); b = self._bin(sigma)
        s = lac_scores(p, y); self.q = {}
        for k in range(self.n_bins):
            m = b == k; sk = s[m]; ck = np.asarray(cluster)[m]
            if not m.any():
                self.q[k] = 1.0; continue
            cnt = pd.Series(ck).map(pd.Series(ck).value_counts()).values; w = 1.0 / cnt
            nc = len(set(ck)); lev = (1 - self.alpha) * (1 + 1 / nc)
            if lev >= 1:
                self.q[k] = 1.0; continue
            o = np.argsort(sk); cw = np.cumsum(w[o]) / w.sum()
            self.q[k] = float(sk[o][np.searchsorted(cw, lev)])
        return self


class LocalisedCP:
    """noise-localised CP: calibration scores weighted by a Gaussian kernel in log(sigma_hat) around the test scan
    (weighted split CP with the test point's own weight placed at +inf; Tibshirani et al. 2019, Guan 2023)"""
    def __init__(self, alpha, h=0.5):
        self.alpha, self.h = alpha, h

    def fit(self, p, y, sigma):
        s = lac_scores(p, y); o = np.argsort(s); self.s = s[o]; self.ls = np.log(np.clip(fin(sigma), 1, 1e4))[o]
        return self

    def thresholds(self, sigma):
        lt = np.log(np.clip(fin(sigma), 1, 1e4)); out = np.empty(len(lt))
        for i in range(0, len(lt), 2000):
            w = np.exp(-0.5 * ((self.ls[None, :] - lt[i:i + 2000, None]) / self.h) ** 2)
            tot = w.sum(1) + 1.0                                    # + test point's own kernel weight K(0)=1
            out[i:i + 2000] = wquantile_sorted(self.s, w / tot[:, None], np.full(len(w), 1 - self.alpha))
        return out

    def predict(self, p, sigma):
        return predict_sets(p, self.thresholds(sigma))


def run_methods(alpha, v, fold):
    cal = P[(P.variant == v) & (P.fold == fold) & (P.split == "cal")]
    c0 = cal[cal.condition == "clean"]; cp = cal[cal.condition.isin(CAL_CONDS)]
    a = (cp.prob.values, cp.label_hard.values, cp.sigma.values)
    M = {"standard": StandardCP(alpha).fit(c0.prob.values, c0.label_hard.values, c0.sigma.values),
         "mondrian_K4": NoiseMondrianCP(alpha, 4).fit(*a),
         "mondrian_K4_cluster": ClusterMondrianCP(alpha, 4).fit_c(*a, cp.uid.values),
         "mondrian_K4_class": NoiseMondrianCP(alpha, 4, class_conditional=True).fit(*a),
         "localised_h0.5": LocalisedCP(alpha, 0.5).fit(*a),
         "localised_h0.25": LocalisedCP(alpha, 0.25).fit(*a),
         "localised_h1.0": LocalisedCP(alpha, 1.0).fit(*a),
         # real-noise reference: noise-Mondrian calibrated on NATIVE scans only (bins of native sigma)
         "mondrian_K4_native_cal": NoiseMondrianCP(alpha, 4).fit(c0.prob.values, c0.label_hard.values, c0.sigma.values)}
    return M


recs = []
for alpha in cfg.get("alphas", [0.10, 0.05]):
    for v in ["dart", "gauss", "baseline"]:
        for fold in sorted(P.fold.unique()):
            te = T[(T.variant == v) & (T.fold == fold) & T.condition.isin(COND_ALL)]
            y = te.label_hard.values.astype(int)
            for m, cpm in run_methods(alpha, v, fold).items():
                s = cpm.predict(te.prob.values, te.sigma.values)
                recs.append(pd.DataFrame(dict(alpha=alpha, variant=v, method=m, fold=fold, uid=te.uid.values,
                                              patient_id=te.patient_id.values, condition=te.condition.values,
                                              sigma=te.sigma.values, y=y, hit=s[np.arange(len(y)), y], size=s.sum(1))))
R = pd.concat(recs, ignore_index=True)
R["stratum"] = pd.Categorical(np.array(STRATA)[np.searchsorted(qe, fin(R.sigma), side="right")], STRATA, ordered=True)
R["unc"] = (R["size"] == 2).astype(float)

# simulated acquisitions (all 9 conditions): coverage by noise stratum
g = R.groupby(["alpha", "variant", "method", "stratum"], observed=True).agg(n=("hit", "size"), coverage=("hit", "mean"),
                                                                           uncertain=("unc", "mean")).reset_index()
g.to_csv(os.path.join(OUT, "cp_repair_by_noise_stratum.csv"), index=False)
o = R.groupby(["alpha", "variant", "method", "condition"]).agg(coverage=("hit", "mean"), set_size=("size", "mean"),
                                                                uncertain=("unc", "mean")).reset_index()
summ = o.groupby(["alpha", "variant", "method"]).agg(worst_condition=("coverage", "min"), mean_coverage=("coverage", "mean"),
                                                     mean_set_size=("set_size", "mean"), uncertain=("uncertain", "mean")).reset_index()
top = R[R.stratum == "top 10%"].groupby(["alpha", "variant", "method"])
summ = summ.merge(top.hit.mean().rename("top10_coverage").reset_index(), on=["alpha", "variant", "method"])
summ = summ.merge(top.unc.mean().rename("top10_uncertain").reset_index(), on=["alpha", "variant", "method"])
ci = []
for (alpha, v, m), d in R[R.stratum == "top 10%"].groupby(["alpha", "variant", "method"]):
    lo, hi = cluster_boot(d, lambda x: x["hit"].mean(), n=500); ci.append(dict(alpha=alpha, variant=v, method=m, top10_lo=lo, top10_hi=hi))
summ = summ.merge(pd.DataFrame(ci), on=["alpha", "variant", "method"])
summ.to_csv(os.path.join(OUT, "cp_repair_summary.csv"), index=False)

gc = R.groupby(["alpha", "variant", "method", "stratum", "y"], observed=True).agg(n=("hit", "size"), coverage=("hit", "mean")).reset_index()
gc["class"] = gc.y.map({0: "benign", 1: "malignant"}); gc.drop(columns="y").to_csv(os.path.join(OUT, "cp_repair_by_noise_and_class.csv"), index=False)
cim = []
for (alpha, v, m), d in R[(R.stratum == "top 10%") & (R.y == 1)].groupby(["alpha", "variant", "method"]):
    lo, hi = cluster_boot(d, lambda x: x["hit"].mean(), n=500)
    cim.append(dict(alpha=alpha, variant=v, method=m, top10_malignant=d.hit.mean(), top10_malignant_lo=lo, top10_malignant_hi=hi, n=len(d)))
pd.DataFrame(cim).to_csv(os.path.join(OUT, "cp_top10_malignant.csv"), index=False)

# real noise: native scans only, strata of the NATIVE noise index
N0 = R[R.condition == "clean"].copy()
qn = np.nanquantile(T[(T.condition == "clean")].drop_duplicates("uid").sigma.values, [.25, .5, .75, .9])
N0["nstratum"] = pd.Categorical(np.array(STRATA)[np.searchsorted(qn, fin(N0.sigma), side="right")], STRATA, ordered=True)
rn = N0.groupby(["alpha", "variant", "method", "nstratum"], observed=True).agg(n=("hit", "size"), coverage=("hit", "mean"),
                                                                              uncertain=("unc", "mean")).reset_index()
cis = []
for (alpha, v, m, st), d in N0[(N0.alpha == 0.1) & N0.method.isin(["standard", "mondrian_K4", "localised_h0.5", "mondrian_K4_class",
                                                                   "mondrian_K4_native_cal"])].groupby(
        ["alpha", "variant", "method", "nstratum"], observed=True):
    lo, hi = cluster_boot(d, lambda x: x["hit"].mean(), n=500); cis.append(dict(alpha=alpha, variant=v, method=m, nstratum=st, lo=lo, hi=hi))
rn = rn.merge(pd.DataFrame(cis), on=["alpha", "variant", "method", "nstratum"], how="left")
rn.to_csv(os.path.join(OUT, "cp_real_native_noise.csv"), index=False)
N0.groupby(["alpha", "variant", "method", "nstratum", "y"], observed=True).agg(n=("hit", "size"), coverage=("hit", "mean")).reset_index().to_csv(
    os.path.join(OUT, "cp_real_native_noise_by_class.csv"), index=False)
pd.DataFrame({"native_sigma_edges_HU": qn}).to_csv(os.path.join(OUT, "cp_real_native_noise_edges.csv"), index=False)

# ---- Fig. 8
MSTY = [("standard", "Standard CP", "#2a78d6", "o", "-"), ("mondrian_K4", "Noise-Mondrian (4 bins)", "#eb6834", "s", "-"),
        ("mondrian_K4_cluster", "Cluster-weighted Mondrian", "#1baf7a", "^", "--"),
        ("localised_h0.5", "Noise-localised CP", "#6250d6", "D", "-"),
        ("mondrian_K4_class", "Noise \u00d7 class Mondrian", "#e87ba4", "v", ":")]
labs = ["Q1", "Q2", "Q3", "Q4", "Top\n10%"]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True)
panels = [(axs[0, 0], "A  DART: all nodules, simulated doses", g[g.variant == "dart"], "stratum"),
          (axs[0, 1], "B  White-noise aug.: all nodules, simulated", g[g.variant == "gauss"], "stratum"),
          (axs[1, 0], "C  DART: malignant nodules only, simulated", gc[(gc.variant == "dart") & (gc.y == 1)], "stratum"),
          (axs[1, 1], "D  DART: native scans only (real noise)", rn[rn.variant == "dart"], "nstratum")]
for ax, ttl, src, key in panels:
    for m, nm, col, mk, ls in MSTY:
        s_ = src[(src.alpha == 0.1) & (src.method == m)].set_index(key).reindex(STRATA)
        if s_.coverage.notna().any():
            ax.plot(range(5), s_.coverage.values, marker=mk, ms=3.5, lw=1.4, ls=ls, color=col, label=nm)
    ax.axhline(0.9, ls="--", lw=0.8, color="#222222"); ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.grid(alpha=1)
    ax.set_xlabel(r"Native-scan noise stratum ($\hat{\sigma}$)" if key == "nstratum" else r"Noise stratum ($\hat{\sigma}$)")
    ax.set_title(ttl, loc="left", fontsize=8.5, fontweight="bold")
for ax in axs[:, 0]:
    ax.set_ylabel("Coverage (target 0.90)")
axs[0, 0].set_ylim(0.66, 0.97)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(FIG, "fig8_conformal_repair.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)
print("B. conformal repairs done")


# ================================================================== C. semantic-feature fragility (DART heads)
D = T[(T.variant == "dart") & T.condition.isin(DOSES)].copy()
srows = []
for k in SEM:
    D[f"r_{k}"] = D.uid.map(NOD[f"sem_{k}"]); D[f"pr_{k}"] = 1 + 4 * D[f"pred_sem_{k}"]
base = D[D.condition == "clean"].set_index("uid")
for k in SEM:
    sd_reader = base[f"r_{k}"].std()
    for c in DOSES:
        d = D[D.condition == c].set_index("uid")
        d = d.assign(delta=d[f"pr_{k}"] - base.loc[d.index, f"pr_{k}"]).reset_index()
        rho = spearman(d[f"pr_{k}"], d[f"r_{k}"])
        lo, hi = cluster_boot(d, lambda x: spearman(x[f"pr_{k}"], x[f"r_{k}"]), n=500)
        dl, dh = cluster_boot(d, lambda x: x.delta.mean(), n=500)
        srows.append(dict(feature=k, condition=c, dose_pct=DOSE_PCT[c], n=len(d), spearman=rho, sp_lo=lo, sp_hi=hi,
                          mean_drift=d.delta.mean(), drift_lo=dl, drift_hi=dh, mean_abs_change=d.delta.abs().mean(),
                          abs_change_vs_reader_sd=d.delta.abs().mean() / sd_reader, reader_sd=sd_reader))
SR = pd.DataFrame(srows); SR.to_csv(os.path.join(OUT, "semantic_dose_fragility.csv"), index=False)
# spiculation call-level check: reader-spiculated (mean >= 2) nodules predicted >= 2 at each dose
spc = []
for c in DOSES:
    d = D[D.condition == c]; rs = d.r_spiculation >= 2
    spc.append(dict(condition=c, dose_pct=DOSE_PCT[c], n_reader_spiculated=int(rs.sum()),
                    sensitivity=float((d.pr_spiculation[rs] >= 2).mean()), specificity=float((d.pr_spiculation[~rs] < 2).mean())))
pd.DataFrame(spc).to_csv(os.path.join(OUT, "semantic_spiculation_calls.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for k, col, mk in zip(SEM, SEM_COL, SEM_MK):
    s = SR[SR.feature == k].sort_values("dose_pct", ascending=False)
    axs[0].errorbar(s.dose_pct, s.spearman, yerr=[s.spearman - s.sp_lo, s.sp_hi - s.spearman], marker=mk, ms=3.5, lw=1.3,
                    capsize=1.5, color=col, label=k.capitalize())
    axs[1].errorbar(s.dose_pct, s.mean_drift, yerr=[s.mean_drift - s.drift_lo, s.drift_hi - s.mean_drift], marker=mk, ms=3.5,
                    lw=1.3, capsize=1.5, color=col, label=k.capitalize())
for ax in axs:
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs]); ax.minorticks_off()
    ax.set_xlabel("Simulated dose"); ax.grid(alpha=1)
axs[0].set_ylabel("Spearman ρ with reader mean"); axs[0].set_title("A  Agreement with radiologists", loc="left", fontsize=8.5, fontweight="bold")
axs[1].axhline(0, color="#555555", lw=0.8)
axs[1].set_ylabel("Mean shift from native\n(rating points, 1–5 scale)"); axs[1].set_title("B  Systematic drift", loc="left", fontsize=8.5, fontweight="bold")
axs[1].legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_semantic_dose.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)
print("C. semantic features done")
print("results ->", OUT)
