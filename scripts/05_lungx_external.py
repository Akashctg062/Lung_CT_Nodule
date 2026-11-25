"""Step 5 - external test on LUNGx (SPIE-AAPM-NCI Lung Nodule Classification Challenge, TCIA).
1) reads the truth spreadsheet(s) and locates every nodule in the DICOM folder,
2) writes QC images (check that the cross-hair is on the nodule!),
3) builds a bank (same acquisition conditions) and
4) evaluates every fold model of every variant (AUC per dose + conformal coverage with the LIDC-calibrated thresholds).
Run:  python scripts/05_lungx_external.py --config configs/default.yaml
"""
import argparse, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dart.utils import load_config, wpath, get_device, SEMANTIC
from dart.dicom_io import find_ct_series, load_series
from dart.bank import build_series_bank
from dart.data import Bank, to_tensor, center_crop
from dart.model import NoduleNet3D
from dart.metrics import auc, cluster_bootstrap
from dart.conformal import NoiseMondrianCP, StandardCP, set_metrics


def col(df, *pats):
    for c in df.columns:
        lc = str(c).lower()
        if all(p in lc for p in pats):
            return c
    return None


def read_truth(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print("  (truth file not found, skipped):", p); continue
        x = pd.read_excel(p)
        cs, cn = col(x, "scan"), col(x, "nodule", "number")
        cxy, cim = col(x, "x,y") or col(x, "position"), col(x, "image")
        cd = col(x, "final", "diagnosis") or col(x, "diagnosis")
        print(f"  {os.path.basename(p)} columns -> scan:{cs!r} nodule:{cn!r} xy:{cxy!r} image:{cim!r} diagnosis:{cd!r}")
        for rr in x.to_dict("records"):
            scan = str(rr.get(cs, "")).strip()
            if not scan or scan.lower() == "nan" or " " in scan or len(scan) > 40 or pd.isna(rr.get(cxy)) or pd.isna(rr.get(cim)):
                continue  # blank rows and the footnotes at the bottom of the sheet
            v = rr[cxy]
            if isinstance(v, (int, float)):  # Excel turned "120, 325" into the number 120325
                v = f"{int(round(v)):,}"
            xy = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", str(v))[:2]]
            if len(xy) < 2:
                print("  could not read x,y for", scan, rr[cxy]); continue
            xy = [xy[0] - 1.0, xy[1] - 1.0]  # sheet uses 1-based pixels: (1,1) = upper-left pixel
            diag = str(rr.get(cd, "")).lower()
            lab = 1 if ("malig" in diag or "cancer" in diag or "carcinoma" in diag) else (0 if "benign" in diag else -1)
            rows.append(dict(scan=scan, nodule=int(rr[cn]) if cn and not pd.isna(rr[cn]) else 1,
                             cx=xy[0], cy=xy[1], image=int(rr[cim]), label_hard=lab, source=os.path.basename(p)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config); dev = get_device(cfg)
    truth = read_truth(cfg["lungx_truth"])
    print(f"LUNGx truth rows: {len(truth)}  (malignant {int((truth.label_hard == 1).sum())}, benign {int((truth.label_hard == 0).sum())})")
    series = find_ct_series(cfg["lungx_root"])
    by_pid = {}
    for suid, s in series.items():
        by_pid.setdefault(s["patient_id"].lower(), []).append((suid, s))
    qc_dir = wpath(cfg, "lungx", "qc", "x"); qc_dir = os.path.dirname(qc_dir)
    bank_dir = os.path.dirname(wpath(cfg, "bank_lungx", "x"))
    rows = []
    for scan, g in truth.groupby("scan"):
        cand = by_pid.get(scan.lower()) or [v for k, vs in by_pid.items() if scan.lower() in k for v in vs]
        if not cand:
            print("  no DICOM found for", scan); continue
        suid, s = max(cand, key=lambda t: len(t[1]["files"]))  # largest CT series of the patient
        ser = load_series(s["files"]); ser["series_uid"] = suid
        inst = np.array(ser["instance_numbers"])
        nods = []
        for r in g.itertuples():
            k = np.where(inst == r.image)[0]
            if not len(k):
                print(f"  {scan}: image number {r.image} not found among InstanceNumbers - skipped"); continue
            zi = int(k[0])  # 'Nodule Center Image' = DICOM InstanceNumber (verified on LUNGx-CT001/002)
            uid = f"{scan}_{r.nodule}"
            nods.append(dict(uid=uid, cx=r.cx, cy=r.cy, cz=float(ser["z_mm"][zi])))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.clip(ser["vol_hu"][zi], -1000, 400), cmap="gray")
            ax.plot(r.cx, r.cy, "r+", ms=25, mew=1); ax.set_title(f"{uid} label={r.label_hard}"); ax.axis("off")
            fig.savefig(os.path.join(qc_dir, f"{uid}.png"), dpi=80); plt.close(fig)
            rows.append(dict(uid=uid, patient_id=scan, label_hard=r.label_hard, cz=nods[-1]["cz"],
                             manufacturer=ser["meta"]["manufacturer"], kernel=ser["meta"]["kernel"],
                             exposure_mAs=ser["meta"]["exposure_mAs"], slice_thickness=ser["meta"]["slice_thickness"]))
        if not all(os.path.exists(os.path.join(bank_dir, f"{n['uid']}.npz")) for n in nods):
            res = build_series_bank(ser, nods, cfg["conditions"], cfg, dev)
            for u, d in res.items():
                np.savez_compressed(os.path.join(bank_dir, f"{u}.npz"), vol=d["vol"], sigma=d["sigma"], names=np.array(d["names"]), version=np.array(2))
    L = pd.DataFrame(rows); L.to_csv(wpath(cfg, "tables", "lungx_nodules.csv"), index=False)
    print(f"Indexed {len(L)} LUNGx nodules. CHECK the QC images in {qc_dir} before trusting results!")
    L = L[L.label_hard >= 0].reset_index(drop=True)
    bank = Bank(bank_dir, L.uid.tolist(), cfg["eval_conditions"], storage="ram")

    # ---- evaluate every fold model ----
    P = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(cfg["work_dir"], "preds", "*.csv"))])
    res, cres = [], []
    for mp in sorted(glob.glob(os.path.join(cfg["work_dir"], "models", "*.pt"))):
        ck = torch.load(mp, map_location=dev, weights_only=False)
        v, fold, T = ck["variant"], ck["fold"], ck["T"]
        model = NoduleNet3D(ck["width"], len(SEMANTIC) if ck["cfg"]["semantic"] > 0 else 0).to(dev)
        model.load_state_dict(ck["state"]); model.eval()
        cal = P[(P.variant == v) & (P.fold == fold) & (P.split == "cal") & (P.label_hard >= 0)]
        cp = cal[cal.condition.isin([c for c in cfg["eval_conditions"] if c == "clean" or c.startswith("d")])]
        c0 = cal[cal.condition == "clean"]
        cps = {"standard": StandardCP(0.1).fit(c0.prob.values, c0.label_hard.values, c0.sigma.values),
               "mondrian": NoiseMondrianCP(0.1, cfg["n_noise_bins"]).fit(cp.prob.values, cp.label_hard.values, cp.sigma.values)}
        for c in cfg["eval_conditions"]:
            with torch.no_grad():
                x = center_crop(to_tensor(bank.get(L.uid.values, [c] * len(L)), dev), cfg["input_size"])
                lg = model(x)[0].float().cpu().numpy()
            p = 1 / (1 + np.exp(-lg / T)); sig = bank.sig(L.uid.values, [c] * len(L))
            res.append(dict(variant=v, fold=fold, condition=c, AUC=auc(L.label_hard, p)))
            for m, cpm in cps.items():
                cres.append(dict(variant=v, fold=fold, condition=c, method=m,
                                 **set_metrics(cpm.predict(p, sig), L.label_hard.values)))
    R = pd.DataFrame(res); C = pd.DataFrame(cres)
    R.to_csv(wpath(cfg, "results", "lungx_auc_per_fold.csv"), index=False)
    C.to_csv(wpath(cfg, "results", "lungx_conformal_per_fold.csv"), index=False)
    s = R.groupby(["variant", "condition"]).AUC.agg(["mean", "std"]).reset_index()
    s.to_csv(wpath(cfg, "results", "lungx_auc_summary.csv"), index=False)
    print(s.pivot(index="variant", columns="condition", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
