"""Step 1 - parse LIDC-IDRI XML annotations, match them to CT series and build the nodule table.
Output: <work_dir>/tables/nodules_all.csv  and  nodules.csv (>= min_readers)
Run:    python scripts/01_index_lidc.py --config configs/default.yaml
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from dart.utils import load_config, wpath, SEMANTIC
from dart.lidc_xml import parse_lidc_xml, cluster_annotations, find_xml_files
from dart.dicom_io import find_ct_series, find_series_dirs, slice_mas


def series_from_folder(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if not f.lower().endswith(".xml")]
    for f in files[:3]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", "") == "CT":
                return str(ds.SeriesInstanceUID), files, ds
        except Exception:
            pass
    return None, files, None


def header_meta(ds):
    def num(t):
        try:
            return float(getattr(ds, t))
        except Exception:
            return np.nan
    ps = [float(v) for v in getattr(ds, "PixelSpacing", [np.nan, np.nan])]
    return dict(patient_id=str(getattr(ds, "PatientID", "")), row_mm=ps[0], col_mm=ps[1],
                manufacturer=str(getattr(ds, "Manufacturer", "")), kernel=str(getattr(ds, "ConvolutionKernel", "")),
                kvp=num("KVP"), tube_current=num("XRayTubeCurrent"), exposure_mAs=num("Exposure"), mAs_eff=slice_mas(ds),
                slice_thickness=num("SliceThickness"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    xml_root = cfg.get("lidc_xml_root") or cfg["lidc_root"]
    if not os.path.isdir(xml_root):
        xml_root = cfg["lidc_root"]
    xmls = find_xml_files(xml_root)
    if xml_root != cfg["lidc_root"]:
        xmls += find_xml_files(cfg["lidc_root"])
    print(f"Found {len(xmls)} XML files")
    parsed = {}
    for p in tqdm(xmls, desc="parse XML"):
        r = parse_lidc_xml(p)
        if r and r["series_uid"] not in parsed:
            parsed[r["series_uid"]] = r
    print(f"{len(parsed)} CT series with LIDC annotations")

    # map series -> files: first try folders that contain the XML, then a full scan for the rest
    series_files, series_ds = {}, {}
    for suid, r in parsed.items():
        s, files, ds = series_from_folder(os.path.dirname(r["path"]))
        if s == suid:
            series_files[suid] = os.path.dirname(r["path"]); series_ds[suid] = ds
    missing = [s for s in parsed if s not in series_files]
    if missing:
        print(f"Indexing series folders under {cfg['lidc_root']} (one header per folder) ...")
        fast = find_series_dirs(cfg["lidc_root"])
        for s in missing:
            if s in fast and fast[s]["modality"] == "CT":
                d = fast[s]["dir"]
                f0 = [f for f in os.listdir(d) if not f.lower().endswith(".xml")][0]
                series_files[s] = d; series_ds[s] = pydicom.dcmread(os.path.join(d, f0), stop_before_pixels=True, force=True)
        missing = [s for s in parsed if s not in series_files]
    if missing:
        print(f"Scanning {cfg['lidc_root']} for {len(missing)} series (one-off, may take a while) ...")
        allser = find_ct_series(cfg["lidc_root"])
        for s in missing:
            if s in allser:
                series_files[s] = os.path.dirname(allser[s]["files"][0])
                series_ds[s] = pydicom.dcmread(allser[s]["files"][0], stop_before_pixels=True, force=True)
    print(f"Matched {len(series_files)}/{len(parsed)} series to DICOM folders")

    rows = []
    for suid, r in tqdm(parsed.items(), desc="cluster"):
        if suid not in series_files:
            continue
        meta = header_meta(series_ds[suid])
        clusters = cluster_annotations(r, (meta["row_mm"], meta["col_mm"]))
        for ci, cl in enumerate(clusters):
            mal = [a["chars"]["malignancy"] for a in cl]
            vote = np.mean([1.0 if m >= 4 else (0.5 if m == 3 else 0.0) for m in mal])
            mean_m = float(np.mean(mal))
            label = 1 if mean_m > 3 else (0 if mean_m < 3 else -1)
            row = dict(uid=f"{meta['patient_id']}_{suid[-8:]}_{ci:02d}", series_uid=suid,
                       series_dir=series_files[suid], n_readers=len(cl), ratings=";".join(map(str, mal)),
                       malig_mean=mean_m, q_soft=float(vote), label_hard=label,
                       cx=float(np.mean([a["cx"] for a in cl])), cy=float(np.mean([a["cy"] for a in cl])),
                       cz=float(np.mean([a["cz"] for a in cl])), diam_mm=float(np.mean([a["diam"] for a in cl])))
            for k in SEMANTIC:
                v = [a["chars"].get(k) for a in cl if a["chars"].get(k)]
                row[f"sem_{k}"] = float(np.mean(v)) if v else np.nan
            row.update(meta)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(wpath(cfg, "tables", "nodules_all.csv"), index=False)
    sel = df[df.n_readers >= cfg["min_readers"]].reset_index(drop=True)
    sel.to_csv(wpath(cfg, "tables", "nodules.csv"), index=False)
    print(f"All nodule clusters: {len(df)} | >= {cfg['min_readers']} readers: {len(sel)} "
          f"(malignant {int((sel.label_hard == 1).sum())}, benign {int((sel.label_hard == 0).sum())}, "
          f"indeterminate {int((sel.label_hard == -1).sum())}) from {sel.patient_id.nunique()} patients")


if __name__ == "__main__":
    main()
