"""Compact 3-D ResNet for nodule malignancy with optional semantic-attribute heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm3d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)), inplace=True)


class NoduleNet3D(nn.Module):
    def __init__(self, width=24, n_semantic=6, dropout=0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv3d(1, w, 3, 1, 1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(Block(w, w), Block(w, w),
                                    Block(w, 2 * w, 2), Block(2 * w, 2 * w),
                                    Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w),
                                    Block(4 * w, 8 * w, 2), Block(8 * w, 8 * w))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(8 * w, 1)
        self.sem = nn.Linear(8 * w, n_semantic) if n_semantic > 0 else None

    def forward(self, x):
        f = self.layers(self.stem(x))
        f = F.adaptive_avg_pool3d(f, 1).flatten(1)
        f = self.drop(f)
        logit = self.head(f).squeeze(1)
        sem = torch.sigmoid(self.sem(f)) if self.sem is not None else None
        return logit, sem
