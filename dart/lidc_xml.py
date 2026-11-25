"""Parser for LIDC-IDRI XML annotation files + cross-reader nodule clustering.

Each LIDC XML (LidcReadMessage) holds up to 4 blinded+unblinded reading sessions.
For nodules >= 3 mm every radiologist gives 9 characteristics (1-5 or 1-6) and an outline per slice.
We keep only <unblindedReadNodule> entries that have <characteristics> with a malignancy rating.
"""
from __future__ import annotations
import os
import re
import xml.etree.ElementTree as ET
import numpy as np

CHARS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin",
         "lobulation", "spiculation", "texture", "malignancy"]


def _strip(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _children(el, name):
    name = name.lower()
    return [c for c in list(el) if _strip(c.tag).lower() == name]


def _child(el, name):
    c = _children(el, name)
    return c[0] if c else None


def _text(el, name, default=None):
    c = _child(el, name)
    return c.text.strip() if (c is not None and c.text) else default


def parse_lidc_xml(path: str) -> dict | None:
    """Return {series_uid, readers: [[annotation,...], ...]} or None if not a CT LIDC file.
    annotation = {reader, nodule_id, chars{...}, rois:[{z, sop_uid, inclusion, xs, ys}]}"""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    if "lidcreadmessage" not in _strip(root.tag).lower():
        return None
    hdr = _child(root, "ResponseHeader")
    suid = None
    if hdr is not None:
        for c in hdr:
            if _strip(c.tag).lower() == "seriesinstanceuid" and c.text:
                suid = c.text.strip()
    if not suid:
        return None
    readers = []
    for ri, sess in enumerate(_children(root, "readingSession")):
        anns = []
        for nod in _children(sess, "unblindedReadNodule"):
            ch = _child(nod, "characteristics")
            if ch is None:
                continue
            chars = {}
            for k in CHARS:
                v = _text(ch, k)
                try:
                    chars[k] = int(v) if v is not None else None
                except ValueError:
                    chars[k] = None
            if not chars.get("malignancy"):
                continue
            rois = []
            for roi in _children(nod, "roi"):
                z = _text(roi, "imageZposition")
                xs, ys = [], []
                for em in _children(roi, "edgeMap"):
                    try:
                        xs.append(float(_text(em, "xCoord"))); ys.append(float(_text(em, "yCoord")))
                    except (TypeError, ValueError):
                        pass
                inc = (_text(roi, "inclusion", "TRUE") or "TRUE").upper() == "TRUE"
                if z is None or not xs:
                    continue
                rois.append(dict(z=float(z), sop_uid=_text(roi, "imageSOP_UID"), inclusion=inc, xs=xs, ys=ys))
            rois_inc = [r for r in rois if r["inclusion"]]
            if not rois_inc:
                continue
            anns.append(dict(reader=ri, nodule_id=_text(nod, "noduleID"), chars=chars, rois=rois_inc))
        readers.append(anns)
    return dict(series_uid=suid, readers=readers, path=path)


def annotation_geometry(ann: dict, spacing_xy) -> dict:
    """Centroid (col=x, row=y in pixels; z in mm) and approximate diameter (mm)."""
    xs = np.concatenate([np.asarray(r["xs"]) for r in ann["rois"]])
    ys = np.concatenate([np.asarray(r["ys"]) for r in ann["rois"]])
    zs = np.array([r["z"] for r in ann["rois"]])
    dx = (xs.max() - xs.min() + 1) * spacing_xy[1]
    dy = (ys.max() - ys.min() + 1) * spacing_xy[0]
    dz = (zs.max() - zs.min()) if len(zs) > 1 else 0.0
    return dict(cx=float(xs.mean()), cy=float(ys.mean()), cz=float(zs.mean()),
                diam=float(max(dx, dy, dz, 3.0)))


def cluster_annotations(parsed: dict, spacing_xy) -> list[list[dict]]:
    """Group annotations of different readers that describe the same physical nodule.
    Two annotations are linked if their centroid distance (mm) < max(3 mm, 0.5*max diameter).
    Union-find (single linkage); at most one annotation per reader is kept per cluster (nearest)."""
    anns = []
    for rd in parsed["readers"]:
        for a in rd:
            a = dict(a); a.update(annotation_geometry(a, spacing_xy)); anns.append(a)
    n = len(anns)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if anns[i]["reader"] == anns[j]["reader"]:
                continue
            d = np.sqrt(((anns[i]["cx"] - anns[j]["cx"]) * spacing_xy[1]) ** 2 +
                        ((anns[i]["cy"] - anns[j]["cy"]) * spacing_xy[0]) ** 2 +
                        (anns[i]["cz"] - anns[j]["cz"]) ** 2)
            if d < max(3.0, 0.5 * max(anns[i]["diam"], anns[j]["diam"])):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(anns[i])
    clusters = []
    for g in groups.values():
        cx = np.mean([a["cx"] for a in g]); cy = np.mean([a["cy"] for a in g]); cz = np.mean([a["cz"] for a in g])
        best = {}
        for a in g:
            d = (a["cx"] - cx) ** 2 + (a["cy"] - cy) ** 2 + (a["cz"] - cz) ** 2
            if a["reader"] not in best or d < best[a["reader"]][0]:
                best[a["reader"]] = (d, a)
        clusters.append([v[1] for v in best.values()])
    return clusters


def find_xml_files(root: str) -> list[str]:
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".xml"):
                out.append(os.path.join(dp, f))
    return out


def patient_id_from_path(path: str) -> str | None:
    m = re.search(r"(LIDC-IDRI-\d{4})", path)
    return m.group(1) if m else None
