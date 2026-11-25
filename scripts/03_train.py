"""Step 3 - patient-level k-fold training of all ablation variants + prediction export.
Output: <work_dir>/models/<variant>_fold<k>.pt
        <work_dir>/preds/<variant>_fold<k>.csv   (val/cal/test predictions for every eval condition)
Run:    python scripts/03_train.py --config configs/default.yaml                 (all variants, all folds)
        python scripts/03_train.py --config configs/default.yaml --variants baseline dart --folds 0
"""
import argparse, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dart.utils import load_config, wpath, set_seed, get_device, make_splits, SEMANTIC
from dart.data import Bank, to_tensor, augment, center_crop
from dart.model import NoduleNet3D
from dart.metrics import auc


def sem_targets(df):
    t = np.stack([(df[f"sem_{k}"].values - 1) / 4.0 for k in SEMANTIC], 1).astype(np.float32)
    return np.clip(t, 0, 1)


@torch.no_grad()
def predict(model, bank, uids, cond, size, dev, bs=64):
    model.eval(); out_l, out_s = [], []
    for i in range(0, len(uids), bs):
        u = uids[i:i + bs]
        x = center_crop(to_tensor(bank.get(u, [cond] * len(u)), dev), size)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            lg, sem = model(x)
        out_l.append(lg.float().cpu().numpy())
        out_s.append(sem.float().cpu().numpy() if sem is not None else np.zeros((len(u), 0)))
    return np.concatenate(out_l), np.concatenate(out_s)


def fit_temperature(logits, y):
    """1-parameter temperature scaling (minimise NLL on validation predictions)."""
    lg = torch.tensor(logits, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    best, bt = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.1), np.log(20), 200)):
        nll = F.binary_cross_entropy_with_logits(lg / T, yt).item()
        if nll < best:
            best, bt = nll, T
    return float(bt)


def train_one(cfg, var_name, var, df, roles, fold, bank, dev):
    set_seed(cfg["seed"] + fold)
    size = cfg["input_size"]
    role = roles[f"fold{fold}"].values
    soft = var["soft_labels"]
    tr = df[(role == "train") & ((df.label_hard >= 0) | soft)]
    va = df[(role == "val") & (df.label_hard >= 0)]
    y_tr = (tr.q_soft.values if soft else tr.label_hard.values).astype(np.float32)
    sem_tr = sem_targets(tr)
    model = NoduleNet3D(width=cfg["width"], n_semantic=len(SEMANTIC) if var["semantic"] > 0 else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    E = cfg["epochs"]; bs = cfg["batch_size"]
    steps = E * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, float(cfg["lr"]), total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and cfg.get("amp", True)))
    dose_conds = cfg["train_dose_conditions"]
    sel_conds = [c for c in ["clean", "d025_r0", "d010_r0"] if c in bank.cidx]
    pos = y_tr.mean(); pw = torch.tensor((1 - pos) / max(pos, 1e-3), device=dev)
    uids = tr.uid.values; best = (-1, None); log = []
    for ep in range(E):
        model.train(); perm = np.random.permutation(len(tr)); tl = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            u = uids[b]; yb = torch.tensor(y_tr[b], device=dev)
            if var["aug"] == "physics":
                ca = [("clean" if np.random.rand() < cfg["p_clean"] else np.random.choice(dose_conds)) for _ in b]
                cb = [np.random.choice(dose_conds) for _ in b]
            else:
                ca = ["clean"] * len(b); cb = ["clean"] * len(b)
            xa = to_tensor(bank.get(u, ca), dev)
            xb = to_tensor(bank.get(u, cb), dev) if var["consistency"] > 0 else None
            if var["aug"] == "gauss":  # control: white noise of matched magnitude, image domain
                lo, hi = cfg["gauss_sigma_hu"]
                s = torch.tensor([0.0 if np.random.rand() < cfg["p_clean"] else np.random.uniform(lo, hi)
                                  for _ in b], device=dev).view(-1, 1, 1, 1, 1) / 1400.0
                xa = xa + s * torch.randn_like(xa)
            views = augment([xa] if xb is None else [xa, xb], size)
            with torch.autocast(dev.type, enabled=dev.type == "cuda" and cfg.get("amp", True)):
                la, sa = model(views[0])
                loss = F.binary_cross_entropy_with_logits(la.float(), yb, pos_weight=pw)
                if xb is not None:
                    lb, sb = model(views[1])
                    loss = 0.5 * (loss + F.binary_cross_entropy_with_logits(lb.float(), yb, pos_weight=pw))
                    pa, pb = torch.sigmoid(la.float()), torch.sigmoid(lb.float())
                    loss = loss + var["consistency"] * ((pa - pb) * (la.float() - lb.float())).mean()  # symmetric KL
                if var["semantic"] > 0:
                    st = torch.tensor(sem_tr[b], device=dev); m = torch.isfinite(st)
                    sem_loss = ((sa.float() - torch.nan_to_num(st)) ** 2 * m).sum() / m.sum().clamp(min=1)
                    if xb is not None:
                        sem_loss = 0.5 * (sem_loss + ((sb.float() - torch.nan_to_num(st)) ** 2 * m).sum() / m.sum().clamp(min=1))
                    loss = loss + var["semantic"] * sem_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(b)
        # model selection: mean val AUC over native + two low-dose acquisitions (same rule for all variants)
        aucs = [auc(va.label_hard.values, predict(model, bank, va.uid.values, c, size, dev)[0]) for c in sel_conds]
        score = float(np.nanmean(aucs))
        log.append(dict(epoch=ep, loss=tl / len(tr), val_auc_sel=score))
        if score > best[0] or (score == best[0] and ep >= E // 2):  # ties: prefer later (trained longer)
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  [{var_name} f{fold}] ep {ep + 1:3d}/{E}  loss {tl / len(tr):.4f}  valAUC(sel) {score:.3f}", flush=True)
    model.load_state_dict(best[1])
    return model, pd.DataFrame(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(cfg)
    df = pd.read_csv(wpath(cfg, "tables", "nodules.csv"))
    bank_dir = os.path.join(cfg["work_dir"], "bank")
    df = df[[os.path.exists(os.path.join(bank_dir, f"{u}.npz")) for u in df.uid]].reset_index(drop=True)
    split_path = wpath(cfg, "tables", "splits.csv")
    if os.path.exists(split_path):
        roles = pd.read_csv(split_path).set_index("uid").loc[df.uid].reset_index(drop=True)
    else:
        roles = make_splits(df, cfg["n_folds"], cfg["seed"])
        out = roles.copy(); out.insert(0, "uid", df.uid.values); out.to_csv(split_path, index=False)
    keep = sorted(set(["clean"] + cfg["train_dose_conditions"] + cfg["eval_conditions"]))
    print(f"Opening bank ({len(df)} nodules, {len(keep)} conditions, storage={cfg.get('bank_storage', 'memmap')}) ...")
    bank = Bank(bank_dir, df.uid.tolist(), keep, storage=cfg.get("bank_storage", "memmap"), cache_dir=cfg["work_dir"])
    variants = a.variants or list(cfg["variants"].keys())
    folds = a.folds if a.folds is not None else list(range(cfg["n_folds"]))
    for fold in folds:
        role = roles[f"fold{fold}"].values
        for vn in variants:
            mp = wpath(cfg, "models", f"{vn}_fold{fold}.pt")
            if os.path.exists(mp) and os.path.exists(wpath(cfg, "preds", f"{vn}_fold{fold}.csv")):
                print(f"skip {vn} fold {fold} (exists)"); continue
            t0 = time.time()
            model, log = train_one(cfg, vn, cfg["variants"][vn], df, roles, fold, bank, dev)
            log.to_csv(wpath(cfg, "logs", f"{vn}_fold{fold}.csv"), index=False)
            # temperature on validation (native + low-dose pooled; same rule for every variant)
            va = df[(role == "val") & (df.label_hard >= 0)]
            lv = np.concatenate([predict(model, bank, va.uid.values, c, cfg["input_size"], dev)[0]
                                 for c in ["clean", "d025_r0", "d010_r0"] if c in bank.cidx])
            yv = np.tile(va.label_hard.values, len(lv) // len(va))
            T = fit_temperature(lv, yv)
            rows = []
            for split in ["val", "cal", "test"]:
                part = df[role == split]
                for c in cfg["eval_conditions"]:
                    if c not in bank.cidx:
                        continue
                    lg, sem = predict(model, bank, part.uid.values, c, cfg["input_size"], dev)
                    sig = bank.sig(part.uid.values, [c] * len(part))
                    d = pd.DataFrame(dict(uid=part.uid.values, patient_id=part.patient_id.values, fold=fold,
                                          variant=vn, split=split, condition=c, logit=lg, T=T,
                                          prob=1 / (1 + np.exp(-lg / T)), sigma=sig,
                                          label_hard=part.label_hard.values, q_soft=part.q_soft.values))
                    for j, k in enumerate(SEMANTIC[:sem.shape[1]]):
                        d[f"pred_sem_{k}"] = sem[:, j]
                    rows.append(d)
            pd.concat(rows).to_csv(wpath(cfg, "preds", f"{vn}_fold{fold}.csv"), index=False)
            torch.save(dict(state=model.state_dict(), T=T, variant=vn, fold=fold, cfg=cfg["variants"][vn],
                            width=cfg["width"]), mp)
            print(f"== {vn} fold {fold} done in {(time.time() - t0) / 60:.1f} min (T={T:.2f})")


if __name__ == "__main__":
    main()
