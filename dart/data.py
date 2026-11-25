"""Acquisition bank (SSD memory-map or RAM) + GPU-side batch augmentation."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import torch

HU_LO, HU_HI = -1000.0, 400.0


class Bank:
    """Acquisition bank for training/evaluation.

    storage='memmap' (default): the bank/<uid>.npz files are unpacked ONCE into one uncompressed file on the
    SSD (<cache_dir>/bank_cache_<n>x<c>.npy, ~4 GB) and read on demand. RAM use stays small; Windows keeps
    recently used parts in its file cache only while memory is free.
    storage='ram': everything is held in RAM (~4 GB for the full LIDC bank).
    """

    def __init__(self, bank_dir: str, uids: list[str], keep: list[str] | None = None, verbose=True,
                 storage: str = "memmap", cache_dir: str | None = None):
        self.uids = list(uids)
        self.uidx = {u: i for i, u in enumerate(self.uids)}
        with np.load(os.path.join(bank_dir, f"{self.uids[0]}.npz"), allow_pickle=False) as z:
            names = [str(n) for n in z["names"]]
            S = z["vol"].shape[-1]
        self.names = names if keep is None else [n for n in names if n in keep]
        self.cidx = {n: i for i, n in enumerate(self.names)}
        n, c = len(self.uids), len(self.names)
        self.sigma = np.full((n, c), np.nan, np.float32)
        if storage == "memmap":
            cache_dir = cache_dir or os.path.dirname(os.path.abspath(bank_dir))
            stem = os.path.join(cache_dir, f"bank_cache_{n}x{c}")
            npy, meta = stem + ".npy", stem + ".json"
            ok = False
            if os.path.exists(npy) and os.path.exists(meta):
                try:
                    m = json.load(open(meta))
                    ok = m.get("uids") == self.uids and m.get("names") == self.names and m.get("complete")
                except Exception:
                    ok = False
            if not ok:
                if verbose:
                    print(f"  building SSD cache {npy} (one-off, a few minutes) ...")
                mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.int16, shape=(n, c, S, S, S))
                sig = np.full((n, c), np.nan, np.float32)
                for i, u in enumerate(self.uids):
                    with np.load(os.path.join(bank_dir, f"{u}.npz"), allow_pickle=False) as z:
                        nm = [str(x) for x in z["names"]]
                        sel = [nm.index(x) for x in self.names]
                        mm[i] = z["vol"][sel]
                        sig[i] = z["sigma"][sel]
                    if verbose and (i + 1) % 200 == 0:
                        print(f"    {i + 1}/{n}")
                mm.flush(); del mm
                np.save(stem + "_sigma.npy", sig)
                json.dump(dict(uids=self.uids, names=self.names, complete=True), open(meta, "w"))
            self.vol = np.load(npy, mmap_mode="r")
            self.sigma = np.load(stem + "_sigma.npy")
            if verbose:
                print(f"  bank on SSD: {npy} ({os.path.getsize(npy) / 1e9:.1f} GB), {n} nodules x {c} conditions")
        else:
            self.vol = np.empty((n, c, S, S, S), np.int16)
            for i, u in enumerate(self.uids):
                with np.load(os.path.join(bank_dir, f"{u}.npz"), allow_pickle=False) as z:
                    nm = [str(x) for x in z["names"]]
                    sel = [nm.index(x) for x in self.names]
                    self.vol[i] = z["vol"][sel]
                    self.sigma[i] = z["sigma"][sel]
                if verbose and (i + 1) % 200 == 0:
                    print(f"  loaded {i + 1}/{n} nodules into RAM")

    def get(self, uids, conds) -> np.ndarray:
        return np.stack([self.vol[self.uidx[u], self.cidx[c]] for u, c in zip(uids, conds)])

    def sig(self, uids, conds) -> np.ndarray:
        return np.array([self.sigma[self.uidx[u], self.cidx[c]] for u, c in zip(uids, conds)], np.float32)


def to_tensor(vols: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(vols.astype(np.float32)).to(device, non_blocking=True)
    x = (x.clamp(HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO)
    return x.unsqueeze(1)  # (B,1,S,S,S)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    s = x.shape[-1]; o = (s - size) // 2
    return x[..., o:o + size, o:o + size, o:o + size]


def augment(xs: list[torch.Tensor], size: int) -> list[torch.Tensor]:
    """Random crop + flips + 90-degree in-plane rotations. The SAME geometric transform is applied
    to every tensor in `xs` (paired views of one nodule under different acquisitions)."""
    B, _, s, _, _ = xs[0].shape
    outs = [[] for _ in xs]
    for i in range(B):
        o = np.random.randint(0, s - size + 1, 3)
        f = np.random.randint(0, 2, 3)
        k = int(np.random.randint(0, 4))
        dims = [d + 2 for d in range(3) if f[d]]
        for j, x in enumerate(xs):
            v = x[i:i + 1, :, o[0]:o[0] + size, o[1]:o[1] + size, o[2]:o[2] + size]
            if dims:
                v = torch.flip(v, dims)
            outs[j].append(torch.rot90(v, k, (3, 4)))
    return [torch.cat(o) for o in outs]
