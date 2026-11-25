"""DICOM series discovery and loading (CT only), returning HU volumes + geometry."""
from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import pydicom


def find_ct_series(root: str, verbose: bool = True) -> dict:
    """Walk a folder tree and group CT DICOM files by SeriesInstanceUID.
    Returns {series_uid: {"files": [...], "patient_id": str, "study_uid": str}}"""
    series = defaultdict(lambda: {"files": [], "patient_id": None, "study_uid": None})
    n = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".xml", ".txt", ".csv", ".json", ".xlsx", ".xls", ".zip")):
                continue
            p = os.path.join(dirpath, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", "") != "CT" or not hasattr(ds, "SeriesInstanceUID"):
                continue
            s = series[str(ds.SeriesInstanceUID)]
            s["files"].append(p)
            s["patient_id"] = str(getattr(ds, "PatientID", ""))
            s["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
            n += 1
            if verbose and n % 20000 == 0:
                print(f"  scanned {n} CT files ...")
    return dict(series)


def slice_mas(ds) -> float:
    """Tube current-time product (mAs) of one slice, robust across vendors.
    1st choice: XRayTubeCurrent (mA, follows tube-current modulation) x ExposureTime (ms) / 1000.
    2nd choice (no ExposureTime): the Exposure tag if plausible (5-1000 mAs) - EXCEPT for GE, whose Exposure
              tag holds unrelated values (e.g. 6 with 401 mA); GE then uses mA x 0.5 s (typical rotation time).
    nan if nothing is usable."""
    def f(tag):
        try:
            v = float(getattr(ds, tag))
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan
    ma, ms, ex = f("XRayTubeCurrent"), f("ExposureTime"), f("Exposure")
    if np.isfinite(ma) and np.isfinite(ms):
        v = ma * ms / 1000.0
        if 1.0 <= v <= 2000.0:
            return v
    is_ge = "GE" in str(getattr(ds, "Manufacturer", "")).upper()
    if not is_ge and np.isfinite(ex) and 5.0 <= ex <= 1000.0:
        return ex
    if np.isfinite(ma) and 5.0 <= ma <= 1500.0:
        return ma * 0.5
    return np.nan


def find_series_dirs(root: str, verbose: bool = True) -> dict:
    """Fast discovery: read ONE header per folder (NBIA / IDC downloads keep one series per folder).
    Returns {series_uid: {"dir": folder, "patient_id": str, "modality": str}}"""
    out, n = {}, 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".xml", ".txt", ".csv", ".json", ".xlsx", ".xls", ".zip", ".tcia")):
                continue
            try:
                ds = pydicom.dcmread(os.path.join(dirpath, f), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if hasattr(ds, "SeriesInstanceUID"):
                out.setdefault(str(ds.SeriesInstanceUID), dict(dir=dirpath, patient_id=str(getattr(ds, "PatientID", "")),
                                                               modality=str(getattr(ds, "Modality", ""))))
                n += 1
                if verbose and n % 200 == 0:
                    print(f"  indexed {n} series folders ...")
                break
    return out


def load_series(files: list[str], series_uid: str | None = None) -> dict:
    """Load a CT series -> dict(vol_hu int16 (Z,H,W), z_mm (Z,), spacing_xy (row,col) mm,
    sop_uids (Z,), instance_numbers (Z,), meta)"""
    slices = []
    for p in files:
        try:
            ds = pydicom.dcmread(p, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "CT") != "CT" or not hasattr(ds, "PixelData"):
            continue
        if series_uid and str(getattr(ds, "SeriesInstanceUID", "")) != series_uid:
            continue
        if hasattr(ds, "ImagePositionPatient"):
            z = float(ds.ImagePositionPatient[2])
        else:
            z = float(getattr(ds, "SliceLocation", 0.0))
        slices.append((z, ds))
    if not slices:
        raise RuntimeError("no image slices")
    slices.sort(key=lambda t: t[0])
    # drop exact duplicates of z
    uniq, seen = [], set()
    for z, ds in slices:
        key = round(z, 3)
        if key in seen:
            continue
        seen.add(key); uniq.append((z, ds))
    shapes = {(int(ds.Rows), int(ds.Columns)) for _, ds in uniq}
    if len(shapes) != 1:
        raise RuntimeError(f"inconsistent slice shapes {shapes}")
    vol = np.empty((len(uniq),) + shapes.pop(), dtype=np.int16)
    for i, (_, ds) in enumerate(uniq):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0)); icpt = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + icpt
        vol[i] = np.clip(hu, -1024, 3071).astype(np.int16)
    ds0 = uniq[0][1]
    ps = [float(v) for v in getattr(ds0, "PixelSpacing", [0.7, 0.7])]

    def num(tag, default=np.nan):
        try:
            return float(getattr(ds0, tag))
        except Exception:
            return default

    meta = dict(
        manufacturer=str(getattr(ds0, "Manufacturer", "")),
        model=str(getattr(ds0, "ManufacturerModelName", "")),
        kernel=str(getattr(ds0, "ConvolutionKernel", "")),
        kvp=num("KVP"), tube_current=num("XRayTubeCurrent"), exposure_mAs=num("Exposure"),
        slice_thickness=num("SliceThickness"), contrast=str(getattr(ds0, "ContrastBolusAgent", "")),
    )
    mas = np.array([slice_mas(ds) for _, ds in uniq], dtype=np.float64)
    return dict(vol_hu=vol, z_mm=np.array([z for z, _ in uniq], dtype=np.float64), mas=mas,
                spacing_xy=(ps[0], ps[1]),
                sop_uids=[str(getattr(ds, "SOPInstanceUID", "")) for _, ds in uniq],
                instance_numbers=[int(getattr(ds, "InstanceNumber", i + 1)) for i, (_, ds) in enumerate(uniq)],
                meta=meta)
