"""Step 2 - simulate every acquisition condition on full slices (GPU) and store 3-D nodule cubes.
Output: <work_dir>/bank/<uid>.npz  +  tables/bank_sigma.csv
Resumable: nodules whose .npz already exists are skipped.
Run:    python scripts/02_build_bank.py --config configs/default.yaml
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from tqdm import tqdm

from dart.utils import load_config, wpath, get_device
from dart.dicom_io import load_series
from dart.bank import build_series_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--table", default=None, help="nodule table (default tables/nodules.csv)")
    ap.add_argument("--bank", default="bank", help="sub-folder of work_dir")
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(cfg)
    df = pd.read_csv(a.table or wpath(cfg, "tables", "nodules.csv"))
    bank_dir = wpath(cfg, a.bank, "x"); bank_dir = os.path.dirname(bank_dir)
    conds = cfg["conditions"]
    BANK_VERSION = 3  # v2: robust noise estimate; v3: photon-starvation cap + GE mAs fix

    import pydicom
    from dart.dicom_io import slice_mas
    _ge_no_et = {}

    def ge_without_exposure_time(series_dir):
        """series whose mAs came from GE's unreliable Exposure tag in v2 -> must be rebuilt"""
        if series_dir not in _ge_no_et:
            try:
                f0 = next(os.path.join(series_dir, f) for f in os.listdir(series_dir) if not f.lower().endswith(".xml"))
                ds = pydicom.dcmread(f0, stop_before_pixels=True, force=True)
                _ge_no_et[series_dir] = ("GE" in str(getattr(ds, "Manufacturer", "")).upper()
                                         and getattr(ds, "ExposureTime", None) in (None, ""))
            except Exception:
                _ge_no_et[series_dir] = True
        return _ge_no_et[series_dir]

    def up_to_date(u, series_dir):
        f = os.path.join(bank_dir, f"{u}.npz")
        if not os.path.exists(f):
            return False
        try:
            with np.load(f) as z:
                ver = int(z["version"]) if "version" in z.files else 1
                if ver >= BANK_VERSION:
                    return True
                if ver < 2:
                    return False
                v, s = z["vol"], z["sigma"]
                sane = v.min() >= -1024 and float((v >= 3071).mean()) < 1e-3 and np.all(np.isfinite(s)) and s.max() < 3000
                if not sane:
                    return False
                return not ge_without_exposure_time(series_dir)
        except Exception:
            return False

    print("checking existing bank files ...")
    todo = df[[not up_to_date(u, d) for u, d in zip(df.uid, df.series_dir)]]
    print(f"{len(df)} nodules, {len(todo)} to process, {len(conds)} conditions, device={dev}")
    t0 = time.time()
    for suid, g in tqdm(list(todo.groupby("series_uid")), desc="series"):
        folder = g.series_dir.iloc[0]
        files = [os.path.join(folder, f) for f in os.listdir(folder) if not f.lower().endswith(".xml")]
        try:
            ser = load_series(files, series_uid=suid)
        except Exception as e:
            print(f"  skip {suid}: {e}"); continue
        ser["series_uid"] = suid
        nods = [dict(uid=r.uid, cx=r.cx, cy=r.cy, cz=r.cz) for r in g.itertuples()]
        res = build_series_bank(ser, nods, conds, cfg, dev)
        for u, d in res.items():
            np.savez_compressed(os.path.join(bank_dir, f"{u}.npz"), vol=d["vol"], sigma=d["sigma"],
                                names=np.array(d["names"]), version=np.array(BANK_VERSION))
    # sigma table
    rows = []
    for u in df.uid:
        p = os.path.join(bank_dir, f"{u}.npz")
        if os.path.exists(p):
            z = np.load(p)
            rows.append(dict(uid=u, **{f"sigma_{n}": float(s) for n, s in zip(z["names"], z["sigma"])}))
    pd.DataFrame(rows).to_csv(wpath(cfg, "tables", f"{a.bank}_sigma.csv"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min; bank has {len(rows)} nodules")


if __name__ == "__main__":
    main()
