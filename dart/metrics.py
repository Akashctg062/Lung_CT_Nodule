"""Discrimination / calibration metrics with patient-clustered bootstrap confidence intervals."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def ece(y, p, n_bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def sens_spec(y, p, thr=0.5):
    y = np.asarray(y); pr = np.asarray(p) >= thr
    sens = (pr & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~pr & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(sens), float(spec)


def cluster_bootstrap(fn, groups, *arrays, n_boot=1000, seed=0):
    """CI of fn(*arrays) resampling whole patients (groups)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        v = fn(*[np.asarray(a)[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_auc_diff(y, p_a, p_b, groups, n_boot=2000, seed=0):
    """AUC(a)-AUC(b) with clustered bootstrap CI and two-sided p-value."""
    d0 = auc(y, p_a) - auc(y, p_b)
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups); ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        a, b = auc(y[idx], p_a[idx]), auc(y[idx], p_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            ds.append(a - b)
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean()) if len(ds) else float("nan")
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(min(p, 1.0))
