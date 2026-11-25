"""Step 4 - all statistics, tables and figures for the paper.
Output: <work_dir>/results/*.csv, *.md, figures/*.png, summary.json
Run:    python scripts/04_evaluate.py --config configs/default.yaml
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dart.utils import load_config, wpath, save_json
from dart.metrics import auc, ece, brier, sens_spec, cluster_bootstrap, paired_auc_diff
from dart.conformal import NoiseMondrianCP, StandardCP, set_metrics

DOSE = {"clean": 1.0, "d050_r0": 0.5, "d025_r0": 0.25, "d010_r0": 0.10, "d005_r0": 0.05}
NICE = {"baseline": "Baseline", "gauss": "+White-noise aug.", "physics": "+Physics aug.",
        "physics_cons": "+Dose consistency", "physics_cons_soft": "+Reader soft labels", "dart": "DART (full)"}
CP_NICE = {"standard": "Standard CP (clean cal.)", "pooled": "Pooled CP (dose-aug. cal.)",
           "mondrian": "Noise-Mondrian CP (proposed)", "mondrian_cc": "Noise×class Mondrian CP (proposed)"}


def md_table(df, path, floatfmt=3):
    with open(path, "w", encoding="utf-8") as f:
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    files = sorted(glob.glob(os.path.join(cfg["work_dir"], "preds", "*.csv")))
    P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    P = P[P.label_hard >= 0]
    out = os.path.join(cfg["work_dir"], "results"); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    variants = [v for v in cfg["variants"] if v in P.variant.unique()]
    conds = [c for c in cfg["eval_conditions"] if c in P.condition.unique()]
    nb = int(cfg.get("n_boot", 1000))
    T = P[P.split == "test"]

    # ---------------- 1. discrimination / calibration ----------------
    rows = []
    for v in variants:
        clean = T[(T.variant == v) & (T.condition == "clean")].set_index(["fold", "uid"]).prob
        for c in conds:
            d = T[(T.variant == v) & (T.condition == c)]
            y, p, g = d.label_hard.values, d.prob.values, d.patient_id.values
            lo, hi = cluster_bootstrap(auc, g, y, p, n_boot=nb)
            se, sp = sens_spec(y, p)
            pc = clean.loc[list(zip(d.fold, d.uid))].values
            flip = float(np.mean((pc >= 0.5) != (p >= 0.5)))
            rows.append(dict(variant=v, condition=c, n=len(d), AUC=auc(y, p), AUC_lo=lo, AUC_hi=hi,
                             ECE=ece(y, p), Brier=brier(y, p), Sens=se, Spec=sp, FlipRate_vs_native=flip,
                             mean_sigma=float(d.sigma.mean())))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "discrimination.csv"), index=False)
    piv = R.pivot(index="variant", columns="condition", values="AUC").loc[variants, conds]
    piv.insert(0, "Model", [NICE.get(v, v) for v in piv.index])
    md_table(piv.reset_index(drop=True), os.path.join(out, "table_auc.md"))
    rob = R[R.condition.isin(["clean", "d005_r0"])].pivot(index="variant", columns="condition", values="AUC")
    if {"clean", "d005_r0"} <= set(rob.columns):
        rob["dose_robustness_index"] = rob["d005_r0"] / rob["clean"]
        rob.to_csv(os.path.join(out, "dose_robustness.csv"))

    # ---------------- 2. paired comparisons ----------------
    comp = []
    for ref in ["baseline", "gauss"]:
        if ref not in variants or "dart" not in variants:
            continue
        for c in conds:
            a = T[(T.variant == "dart") & (T.condition == c)].set_index(["fold", "uid"]).sort_index()
            b = T[(T.variant == ref) & (T.condition == c)].set_index(["fold", "uid"]).sort_index()
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx], b.loc[idx]
            d0, lo, hi, p = paired_auc_diff(a.label_hard.values, a.prob.values, b.prob.values,
                                            a.patient_id.values, n_boot=max(nb, 1000))
            comp.append(dict(comparison=f"DART - {ref}", condition=c, dAUC=d0, lo=lo, hi=hi, p=p))
    pd.DataFrame(comp).to_csv(os.path.join(out, "paired_auc_tests.csv"), index=False)

    # ---------------- 3. conformal prediction ----------------
    cal_conds = [c for c in conds if c == "clean" or c.startswith("d")]
    crow = []
    for alpha in cfg["alphas"]:
        for v in variants:
            for fold in sorted(P.fold.unique()):
                cal = P[(P.variant == v) & (P.fold == fold) & (P.split == "cal")]
                te = P[(P.variant == v) & (P.fold == fold) & (P.split == "test")]
                if cal.empty or te.empty:
                    continue
                c0 = cal[cal.condition == "clean"]; cp = cal[cal.condition.isin(cal_conds)]
                methods = {
                    "standard": StandardCP(alpha).fit(c0.prob.values, c0.label_hard.values, c0.sigma.values),
                    "pooled": StandardCP(alpha).fit(cp.prob.values, cp.label_hard.values, cp.sigma.values),
                    "mondrian": NoiseMondrianCP(alpha, cfg["n_noise_bins"]).fit(cp.prob.values, cp.label_hard.values, cp.sigma.values),
                    "mondrian_cc": NoiseMondrianCP(alpha, cfg["n_noise_bins"], True).fit(cp.prob.values, cp.label_hard.values, cp.sigma.values),
                }
                for m, cpm in methods.items():
                    for c in conds:
                        d = te[te.condition == c]
                        sets = cpm.predict(d.prob.values, d.sigma.values)
                        crow.append(dict(alpha=alpha, variant=v, fold=fold, method=m, condition=c,
                                         _sets=sets, _y=d.label_hard.values))
    # pool test points over folds, then compute metrics
    C = pd.DataFrame(crow)
    agg = []
    for (alpha, v, m, c), g in C.groupby(["alpha", "variant", "method", "condition"]):
        sets = np.concatenate(g._sets.values); y = np.concatenate(g._y.values)
        agg.append(dict(alpha=alpha, variant=v, method=m, condition=c, n=len(y), **set_metrics(sets, y)))
    CA = pd.DataFrame(agg); CA.to_csv(os.path.join(out, "conformal.csv"), index=False)
    for alpha in cfg["alphas"]:
        for v in ["dart", "baseline"]:
            sub = CA[(CA.alpha == alpha) & (CA.variant == v)]
            if sub.empty:
                continue
            t = sub.pivot(index="method", columns="condition", values="coverage")[conds]
            t["worst_case_coverage"] = t.min(axis=1)
            s = sub.pivot(index="method", columns="condition", values="set_size")[conds].mean(axis=1)
            t["mean_set_size"] = s
            t.insert(0, "Method", [CP_NICE.get(i, i) for i in t.index])
            md_table(t.reset_index(drop=True), os.path.join(out, f"table_conformal_{v}_alpha{alpha}.md"))

    # ---------------- 4. native exposure (real-world dose variation inside LIDC) ----------------
    nod = pd.read_csv(wpath(cfg, "tables", "nodules.csv"))
    mcol = "mAs_eff" if "mAs_eff" in nod.columns else "tube_current"
    erows = []
    per_nod = nod.set_index("uid")[mcol]
    tested = per_nod.loc[per_nod.index.intersection(T.uid.unique())].dropna()
    e1, e2 = np.quantile(tested.values, [1 / 3, 2 / 3]) if len(tested) else (0, 0)  # same edges for every variant
    if len(tested) > 30 and e2 > e1:
        ex = T[T.condition == "clean"].copy()
        ex["mAs"] = ex.uid.map(per_nod)
        ex["tertile"] = pd.cut(ex.mAs, [-np.inf, e1, e2, np.inf], labels=["low", "mid", "high"])
        for v in variants:
            for t_, g in ex[ex.variant == v].groupby("tertile", observed=True):
                lo, hi = cluster_bootstrap(auc, g.patient_id.values, g.label_hard.values, g.prob.values, n_boot=nb)
                erows.append(dict(variant=v, mAs_tertile=t_, mAs_range=f"{g.mAs.min():.0f}-{g.mAs.max():.0f}",
                                  n=len(g), AUC=auc(g.label_hard, g.prob), AUC_lo=lo, AUC_hi=hi))
    pd.DataFrame(erows).to_csv(os.path.join(out, "native_exposure_tertiles.csv"), index=False)

    # ---------------- 4b. paired decision-flip comparison (native -> low dose) ----------------
    frows = []
    for ref in ["baseline", "gauss"]:
        if ref not in variants or "dart" not in variants:
            continue
        for c in [x for x in conds if x != "clean"]:
            def flips(v):
                a = T[(T.variant == v) & (T.condition == "clean")].set_index(["fold", "uid"]).sort_index()
                b = T[(T.variant == v) & (T.condition == c)].set_index(["fold", "uid"]).sort_index()
                return ((a.prob >= .5) != (b.prob >= .5)).astype(float), a.patient_id
            fd, g = flips("dart"); fr, _ = flips(ref)
            diff = (fd - fr.loc[fd.index]).values; gr = g.values
            rng = np.random.default_rng(0); ug = np.unique(gr); idx = {k: np.where(gr == k)[0] for k in ug}
            bs = [diff[np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])].mean() for _ in range(max(nb, 1000))]
            bs = np.array(bs)
            frows.append(dict(comparison=f"DART - {ref}", condition=c, flip_dart=fd.mean(), flip_ref=fr.mean(),
                              diff=diff.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                              p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    pd.DataFrame(frows).to_csv(os.path.join(out, "paired_flip_tests.csv"), index=False)

    # ---------------- 4c. conformal coverage conditional on the noise level ----------------
    # (a) strata of the blind noise index over ALL test points; (b) each method's OWN bins; (c) K sensitivity
    srows, own_rows = [], []
    allsig = T.drop_duplicates(["uid", "condition"]).sigma.values
    qe = np.nanquantile(allsig, [.25, .5, .75, .9])
    labels = ["Q1 lowest", "Q2", "Q3", "Q4 (75-90%)", "top 10%"]
    for alpha in cfg["alphas"]:
        for v in variants:
            for K in sorted(set([1, 2, int(cfg["n_noise_bins"]), 8])):
                for fold in sorted(P.fold.unique()):
                    cal = P[(P.variant == v) & (P.fold == fold) & (P.split == "cal")]
                    te = P[(P.variant == v) & (P.fold == fold) & (P.split == "test")]
                    if cal.empty or te.empty:
                        continue
                    cpool = cal[cal.condition.isin(cal_conds)]
                    c0 = cal[cal.condition == "clean"]
                    meths = {f"mondrian_K{K}": NoiseMondrianCP(alpha, K).fit(cpool.prob.values, cpool.label_hard.values, cpool.sigma.values)}
                    if K == 1:
                        meths["standard"] = StandardCP(alpha).fit(c0.prob.values, c0.label_hard.values, c0.sigma.values)
                    for m, cpm in meths.items():
                        sets = cpm.predict(te.prob.values, te.sigma.values)
                        y = te.label_hard.values.astype(int)
                        cov = sets[np.arange(len(y)), y]; unc = sets.sum(1) == 2
                        sig = np.where(np.isfinite(te.sigma.values), te.sigma.values, np.inf)
                        stratum = np.searchsorted(qe, sig, side="right")
                        for k in range(5):
                            msk = stratum == k
                            srows.append(dict(alpha=alpha, variant=v, method=m, stratum=labels[k], n=int(msk.sum()),
                                              cov_sum=float(cov[msk].sum()), unc_sum=float(unc[msk].sum())))
                        if m.startswith("mondrian") and K > 1:
                            b = cpm._bin(te.sigma.values)
                            for k in range(K):
                                msk = b == k
                                own_rows.append(dict(alpha=alpha, variant=v, method=m, own_bin=k, n=int(msk.sum()),
                                                     cov_sum=float(cov[msk].sum())))
    S = pd.DataFrame(srows).groupby(["alpha", "variant", "method", "stratum"], sort=False)[["n", "cov_sum", "unc_sum"]].sum().reset_index()
    S["coverage"] = S.cov_sum / S.n; S["uncertain_rate"] = S.unc_sum / S.n
    S.drop(columns=["cov_sum", "unc_sum"]).to_csv(os.path.join(out, "conformal_by_noise_stratum.csv"), index=False)
    O = pd.DataFrame(own_rows).groupby(["alpha", "variant", "method", "own_bin"])[["n", "cov_sum"]].sum().reset_index()
    O["coverage"] = O.cov_sum / O.n
    O.drop(columns=["cov_sum"]).to_csv(os.path.join(out, "conformal_by_own_bin.csv"), index=False)
    for alpha in cfg["alphas"]:
        for v in [x for x in ["dart", "gauss", "baseline"] if x in variants]:
            t = S[(S.alpha == alpha) & (S.variant == v)].pivot(index="method", columns="stratum", values="coverage")[labels]
            t.insert(0, "Method", t.index)
            md_table(t.reset_index(drop=True), os.path.join(out, f"table_conformal_by_noise_{v}_alpha{alpha}.md"))
    edges_note = dict(noise_strata_edges_HU=[float(x) for x in qe])

    # ---------------- 5. figures ----------------
    dconds = [c for c in DOSE if c in conds]
    xs = [DOSE[c] * 100 for c in dconds]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for v in variants:
        r = R[(R.variant == v)].set_index("condition").loc[dconds]
        ln, = ax.plot(xs, r.AUC, marker="o", label=NICE.get(v, v), lw=2.4 if v == "dart" else 1.4)
        if v in ("dart", "baseline"):  # band in the same colour as its line
            ax.fill_between(xs, r.AUC_lo, r.AUC_hi, alpha=0.15, color=ln.get_color(), lw=0)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}%" for x in xs])
    ax.set_xlabel("Simulated dose (% of native acquisition)"); ax.set_ylabel("AUC (test folds pooled)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig2_auc_vs_dose.png"), dpi=800); plt.close(fig)

    for alpha in cfg["alphas"]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for m in ["standard", "pooled", "mondrian", "mondrian_cc"]:
            s = CA[(CA.alpha == alpha) & (CA.variant == "dart") & (CA.method == m)].set_index("condition")
            if s.empty:
                continue
            s = s.loc[dconds]
            axs[0].plot(xs, s.coverage, marker="o", label=CP_NICE[m])
            axs[1].plot(xs, s.uncertain_rate, marker="o", label=CP_NICE[m])
        axs[0].axhline(1 - alpha, ls="--", c="k", lw=1); axs[0].set_ylabel("Empirical coverage")
        axs[1].set_ylabel("Fraction flagged 'uncertain' (set = {benign, malignant})")
        for a_ in axs:
            a_.set_xscale("log"); a_.invert_xaxis(); a_.set_xticks(xs); a_.set_xticklabels([f"{x:g}%" for x in xs])
            a_.set_xlabel("Simulated dose"); a_.grid(alpha=.3)
        axs[0].legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"fig3_conformal_alpha{alpha}.png"), dpi=800); plt.close(fig)

    d = T[(T.variant == variants[-1])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([d[d.condition == c].sigma.dropna() for c in conds], showfliers=False)
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Blind noise estimate σ̂ (HU)"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig4_sigma_by_condition.png"), dpi=800); plt.close(fig)

    # reliability diagram native vs 5 %
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for v in ["baseline", "dart"]:
        for c in ["clean", "d005_r0"]:
            dd = T[(T.variant == v) & (T.condition == c)]
            if dd.empty:
                continue
            b = np.clip((dd.prob * 10).astype(int), 0, 9)
            mp = dd.groupby(b).prob.mean(); fr = dd.groupby(b).label_hard.mean()
            ax.plot(mp, fr, marker="o", label=f"{NICE.get(v, v)} – {c}")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig5_reliability.png"), dpi=800); plt.close(fig)

    summ = dict(**edges_note, n_test_points=int(T[(T.variant == variants[0]) & (T.condition == "clean")].shape[0]),
                n_patients=int(T.patient_id.nunique()),
                auc={f"{a}|{b}": float(x) for (a, b), x in R.set_index(["variant", "condition"]).AUC.items()})
    save_json(summ, os.path.join(out, "summary.json"))
    print(piv.to_string()); print("\nResults written to", out)


if __name__ == "__main__":
    main()
