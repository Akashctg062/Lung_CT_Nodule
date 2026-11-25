"""Create a tiny synthetic LIDC-like dataset (DICOM + LIDC XML) to smoke-test the whole pipeline
without downloading anything.  python tests/make_synthetic_lidc.py --out <folder> --patients 24"""
import argparse, os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage

NS = "http://www.nih.gov"


def write_slice(path, arr_hu, z, pid, study, series, inst, spacing):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "CT"; ds.PatientID = pid; ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID; ds.SOPClassUID = CTImageStorage
    ds.InstanceNumber = inst; ds.ImagePositionPatient = [-100.0, -100.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.PixelSpacing = [spacing, spacing]
    ds.SliceThickness = 2.5; ds.Manufacturer = "SYNTH"; ds.ConvolutionKernel = "STANDARD"
    ds.KVP = 120; ds.Exposure = int(np.random.choice([40, 120, 300])); ds.XRayTubeCurrent = 100
    ds.Rows, ds.Columns = arr_hu.shape; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
    ds.PixelRepresentation = 1; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = 0; ds.RescaleSlope = 1
    ds.PixelData = arr_hu.astype(np.int16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds.SOPInstanceUID


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--patients", type=int, default=24)
    a = ap.parse_args(); rng = np.random.default_rng(1)
    N, Z, sp = 160, 40, 1.25
    for p in range(1, a.patients + 1):
        pid = f"LIDC-IDRI-{p:04d}"; study, series = generate_uid(), generate_uid()
        d = os.path.join(a.out, pid, "study1", "series1"); os.makedirs(d, exist_ok=True)
        yy, xx = np.mgrid[:N, :N] - N / 2
        body = (xx / 70) ** 2 + (yy / 52) ** 2 < 1
        lung = ((xx - 30) / 24) ** 2 + (yy / 36) ** 2 < 1
        lung |= ((xx + 30) / 24) ** 2 + (yy / 36) ** 2 < 1
        vol = np.full((Z, N, N), -1000.0); vol[:, body] = 40; vol[:, lung] = -850
        nods = []
        for k in range(2):
            mal = rng.random() < 0.45
            r = rng.uniform(5, 9) if mal else rng.uniform(2.5, 5)  # radius in pixels
            cx = N / 2 + (30 if k == 0 else -30) + rng.uniform(-6, 6); cy = N / 2 + rng.uniform(-15, 15); cz = rng.uniform(14, Z - 14)
            zz, yv, xv = np.mgrid[:Z, :N, :N]
            ang = np.arctan2(yv - cy, xv - cx)
            rad = r * (1 + (0.35 * np.sin(7 * ang) if mal else 0))  # spiculated-ish border for malignant
            m = ((xv - cx) ** 2 + (yv - cy) ** 2 + ((zz - cz) * 2.5 / sp) ** 2) < rad ** 2
            vol[m] = 30
            nods.append((cx, cy, cz, r, mal))
        vol += rng.normal(0, 12, vol.shape)
        z0 = -150.0; sops = []
        for i in range(Z):
            sops.append(write_slice(os.path.join(d, f"{i + 1:04d}.dcm"), vol[i], z0 + i * 2.5, pid, study, series, i + 1, sp))
        # XML with 4 readers
        xml = [f'<?xml version="1.0" encoding="UTF-8"?>\n<LidcReadMessage xmlns="{NS}">',
               f"<ResponseHeader><SeriesInstanceUid>{series}</SeriesInstanceUid><StudyInstanceUID>{study}</StudyInstanceUID></ResponseHeader>"]
        for rd in range(4):
            xml.append("<readingSession><servicingRadiologistID>r%d</servicingRadiologistID>" % rd)
            for j, (cx, cy, cz, r, mal) in enumerate(nods):
                score = int(np.clip(round((4.3 if mal else 1.8) + rng.normal(0, 0.8)), 1, 5))
                xml.append(f"<unblindedReadNodule><noduleID>n{j}</noduleID><characteristics>"
                           f"<subtlety>4</subtlety><internalStructure>1</internalStructure><calcification>6</calcification>"
                           f"<sphericity>{3 if mal else 5}</sphericity><margin>{2 if mal else 5}</margin><lobulation>{3 if mal else 1}</lobulation>"
                           f"<spiculation>{4 if mal else 1}</spiculation><texture>5</texture><malignancy>{score}</malignancy></characteristics>")
                for zi in range(int(cz - r * sp / 2.5), int(cz + r * sp / 2.5) + 1):
                    if 0 <= zi < Z:
                        xml.append(f"<roi><imageZposition>{z0 + zi * 2.5:.1f}</imageZposition><imageSOP_UID>{sops[zi]}</imageSOP_UID><inclusion>TRUE</inclusion>")
                        for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                            xml.append(f"<edgeMap><xCoord>{int(cx + r * np.cos(t) + rng.normal(0, .5))}</xCoord><yCoord>{int(cy + r * np.sin(t))}</yCoord></edgeMap>")
                        xml.append("</roi>")
                xml.append("</unblindedReadNodule>")
            xml.append("</readingSession>")
        xml.append("</LidcReadMessage>")
        open(os.path.join(d, "069.xml"), "w").write("\n".join(xml))
    print("synthetic dataset written to", a.out)


if __name__ == "__main__":
    main()
