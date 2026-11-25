from __future__ import annotations
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

SEMANTIC = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    os.makedirs(cfg["work_dir"], exist_ok=True)
    return cfg


def wpath(cfg, *parts) -> str:
    p = os.path.join(cfg["work_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device(cfg=None):
    want = (cfg or {}).get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def make_splits(df: pd.DataFrame, n_folds: int, seed: int, val_frac=0.15, cal_frac=0.15) -> pd.DataFrame:
    """Patient-level splits. Each nodule gets, for every fold k, a role in {train,val,cal,test}."""
    y = df["label_hard"].replace(-1, 2).values  # stratify on 3 groups (benign, malignant, indeterminate)
    g = df["patient_id"].values
    roles = pd.DataFrame(index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr_idx, te_idx) in enumerate(sgkf.split(df, y, g)):
        role = np.array(["train"] * len(df), dtype=object)
        role[te_idx] = "test"
        rest = df.iloc[tr_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac + cal_frac, random_state=seed + k)
        a, b = next(gss.split(rest, groups=rest["patient_id"]))
        hold = rest.iloc[b]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=cal_frac / (val_frac + cal_frac), random_state=seed + 100 + k)
        v, c = next(gss2.split(hold, groups=hold["patient_id"]))
        role[df.index.get_indexer(hold.index[v])] = "val"
        role[df.index.get_indexer(hold.index[c])] = "cal"
        roles[f"fold{k}"] = role
    return roles


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
