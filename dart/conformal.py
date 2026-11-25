"""Split conformal prediction for binary malignancy with LAC scores, including the proposed
noise-conditional (Mondrian) variant whose groups are bins of the blind noise estimate sigma_hat.

Guarantee (Vovk 2003, Mondrian CP): if calibration and test points that fall in the same
noise bin are exchangeable, P(y in C(x) | bin) >= 1 - alpha for every bin, hence coverage is
maintained at every dose level whose sigma_hat lands in that bin - without knowing the dose.
"""
from __future__ import annotations
import numpy as np


def lac_scores(p_mal: np.ndarray, y: np.ndarray) -> np.ndarray:
    p_y = np.where(y == 1, p_mal, 1 - p_mal)
    return 1.0 - p_y


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return 1.0  # not enough calibration points -> trivial (full) set
    return float(np.sort(scores)[k - 1])


def predict_sets(p_mal: np.ndarray, qhat) -> np.ndarray:
    """returns (n,2) boolean membership [benign, malignant]; qhat scalar or per-sample array"""
    q = np.broadcast_to(np.asarray(qhat, float), p_mal.shape)
    return np.stack([(p_mal) <= q, (1 - p_mal) <= q], 1)  # score(benign)=p_mal, score(mal)=1-p_mal


class NoiseMondrianCP:
    def __init__(self, alpha=0.1, n_bins=4, class_conditional=False):
        self.alpha, self.n_bins, self.cc = alpha, n_bins, class_conditional

    @staticmethod
    def _clean(sigma):
        """a noise index that could not be measured (NaN: no usable soft tissue in an extremely noisy
        scan) is treated as 'noisiest' -> highest bin"""
        s = np.asarray(sigma, float)
        return np.where(np.isfinite(s), s, np.inf)

    def fit(self, p_mal, y, sigma):
        sigma = self._clean(sigma)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        fin = sigma[np.isfinite(sigma)]
        self.edges = np.quantile(fin, qs) if (self.n_bins > 1 and fin.size) else np.array([])
        b = self._bin(sigma)
        s = lac_scores(p_mal, y)
        self.q = {}
        for k in range(self.n_bins):
            if self.cc:
                for c in (0, 1):
                    m = (b == k) & (y == c)
                    self.q[(k, c)] = conformal_quantile(s[m], self.alpha)
            else:
                self.q[k] = conformal_quantile(s[b == k], self.alpha)
        return self

    def _bin(self, sigma):
        return np.searchsorted(self.edges, self._clean(sigma), side="right")

    def predict(self, p_mal, sigma):
        b = self._bin(sigma)
        if not self.cc:
            q = np.array([self.q[k] for k in b])
            return predict_sets(p_mal, q)
        q0 = np.array([self.q[(k, 0)] for k in b]); q1 = np.array([self.q[(k, 1)] for k in b])
        return np.stack([p_mal <= q0, (1 - p_mal) <= q1], 1)


class StandardCP(NoiseMondrianCP):
    def __init__(self, alpha=0.1, class_conditional=False):
        super().__init__(alpha, 1, class_conditional)


def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    cover = sets[np.arange(len(y)), y.astype(int)]
    size = sets.sum(1)
    single = size == 1
    pred = np.where(sets[:, 1] & ~sets[:, 0], 1, 0)
    return dict(coverage=float(cover.mean()), set_size=float(size.mean()),
                uncertain_rate=float((size == 2).mean()), empty_rate=float((size == 0).mean()),
                singleton_acc=float((pred[single] == y[single]).mean()) if single.any() else float("nan"),
                coverage_malignant=float(cover[y == 1].mean()) if (y == 1).any() else float("nan"),
                coverage_benign=float(cover[y == 0].mean()) if (y == 0).any() else float("nan"))
