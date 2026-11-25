"""
CT physics utilities: parallel-beam Radon transform, filtered back-projection (FBP)
and projection-domain low-dose noise insertion, all in PyTorch so they run on the GPU.

Noise-insertion model (Yu et al., Med Phys 2012; Zeng et al., 2015):
    For a scan acquired with incident flux I0 per ray and line integral p,
    simulating a dose fraction a (0<a<=1) requires adding zero-mean noise to p with variance
        var_add = (1-a)/a * exp(p)/I0 * (1 + (1+a)/a * sigma_e^2 * exp(p)/I0)
    where sigma_e^2 is the electronic-noise variance (in photon units).
    The noise sinogram is reconstructed with FBP and added to the original image,
    so anatomy is preserved exactly and only quantum + electronic noise changes.

Simplifications (reported as limitations in the paper):
    * parallel-beam geometry instead of fan/helical; no bowtie filter; no tube-current modulation;
    * effective I0 is a single calibrated constant per dataset (see scripts/06_validate_noise_sim.py).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

MU_WATER = 0.0192  # linear attenuation of water, mm^-1 (~70 keV effective energy)


def hu_to_mu(hu: torch.Tensor) -> torch.Tensor:
    return torch.clamp(MU_WATER * (1.0 + hu / 1000.0), min=0.0)


def _ramp_filter(n_det: int, window: str, device, dtype):
    """Frequency-domain ramp filter (length = padded detector size) with apodisation window."""
    n_pad = 2 ** int(math.ceil(math.log2(2 * n_det)))
    # spatial-domain ramp (Kak & Slaney) gives correct DC behaviour
    n = torch.arange(-(n_pad // 2), n_pad // 2, device=device, dtype=dtype)
    h = torch.zeros_like(n)
    h[n == 0] = 0.25
    odd = (n % 2) != 0
    h[odd] = -1.0 / (math.pi * n[odd]) ** 2
    H = torch.fft.fft(torch.fft.ifftshift(h)).real.abs()
    f = torch.fft.fftfreq(n_pad, device=device).to(dtype)  # cycles/pixel in [-0.5,0.5)
    w = window.lower()
    if w in ("ramlak", "ram-lak", "sharp"):
        win = torch.ones_like(f)
    elif w in ("shepp-logan", "shepplogan", "standard"):
        x = math.pi * f
        win = torch.where(f == 0, torch.ones_like(f), torch.sin(x) / x)
    elif w in ("cosine",):
        win = torch.cos(math.pi * f)
    elif w in ("hann", "hanning", "smooth"):
        win = 0.5 * (1 + torch.cos(2 * math.pi * f))
    else:
        raise ValueError(f"unknown FBP window {window}")
    return H * win, n_pad


class ParallelBeamCT:
    """Rotation-based parallel-beam projector / FBP for square images of size N.

    The image is zero-padded to M = ceil(N*sqrt(2)) so that no ray leaves the field of view.
    Units: forward projection returns sums of pixel values (multiply by pixel size in mm to obtain
    line integrals when the image holds mu in mm^-1).
    """

    def __init__(self, n: int, n_angles: int = 360, device="cpu", dtype=torch.float32, angle_chunk: int = 45):
        self.n = n
        self.m = int(math.ceil(n * math.sqrt(2)))
        if (self.m - n) % 2:
            self.m += 1
        self.pad = (self.m - n) // 2
        self.n_angles = n_angles
        self.device = torch.device(device)
        self.dtype = dtype
        self.angle_chunk = angle_chunk
        self.theta = torch.arange(n_angles, device=self.device, dtype=dtype) * (math.pi / n_angles)
        self._filters = {}
        self.scale = 1.0
        self.scale = self._calibrate_scale()

    # ---------- geometry helpers ----------
    def _rot_grid(self, theta: torch.Tensor, size: int):
        c, s = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros_like(c)
        A = torch.stack([torch.stack([c, -s, zeros], -1), torch.stack([s, c, zeros], -1)], -2)  # (k,2,3)
        return F.affine_grid(A, (theta.numel(), 1, size, size), align_corners=False)

    def forward_project(self, img: torch.Tensor) -> torch.Tensor:
        """img (B,N,N) -> sinogram (B,A,M) as sums of pixel values along rays."""
        B = img.shape[0]
        x = F.pad(img, (self.pad, self.pad, self.pad, self.pad)).unsqueeze(1)  # (B,1,M,M)
        out = []
        for i in range(0, self.n_angles, self.angle_chunk):
            th = self.theta[i:i + self.angle_chunk]
            k = th.numel()
            grid = self._rot_grid(th, self.m)  # (k,M,M,2)
            xr = F.grid_sample(x.repeat_interleave(k, 0), grid.repeat(B, 1, 1, 1),
                               mode="bilinear", padding_mode="zeros", align_corners=False)
            out.append(xr.sum(dim=2).view(B, k, self.m))  # sum over rows -> projection along y
        return torch.cat(out, 1)

    def _filter(self, sino: torch.Tensor, window: str):
        key = window.lower()
        if key not in self._filters:
            self._filters[key] = _ramp_filter(self.m, key, self.device, self.dtype)
        H, n_pad = self._filters[key]
        S = torch.fft.fft(F.pad(sino, (0, n_pad - self.m)), dim=-1)
        return torch.fft.ifft(S * H, dim=-1).real[..., : self.m]

    def back_project(self, q: torch.Tensor) -> torch.Tensor:
        """Filtered sinogram (B,A,M) -> image (B,N,N) via smearing + rotation."""
        B = q.shape[0]
        acc = torch.zeros(B, 1, self.m, self.m, device=q.device, dtype=q.dtype)
        for i in range(0, self.n_angles, self.angle_chunk):
            th = self.theta[i:i + self.angle_chunk]
            k = th.numel()
            smear = q[:, i:i + k, :].reshape(B * k, 1, 1, self.m).expand(B * k, 1, self.m, self.m)
            grid = self._rot_grid(-th, self.m).repeat(B, 1, 1, 1)
            r = F.grid_sample(smear, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            acc += r.view(B, k, self.m, self.m).sum(1, keepdim=True)
        img = acc[:, 0, self.pad:self.pad + self.n, self.pad:self.pad + self.n]
        return img * (math.pi / self.n_angles) * self.scale

    def fbp(self, sino: torch.Tensor, window: str = "shepp-logan") -> torch.Tensor:
        return self.back_project(self._filter(sino, window))

    @torch.no_grad()
    def _calibrate_scale(self) -> float:
        """Numerically fix the FBP gain so that fbp(R(x)) ~= x for a smooth disk phantom."""
        n = self.n
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, n, device=self.device),
                                torch.linspace(-1, 1, n, device=self.device), indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2)
        ph = torch.clamp((0.6 - r) / 0.05, 0, 1).to(self.dtype)[None]
        rec = self.fbp(self.forward_project(ph), "ramlak")
        mask = r < 0.5
        return float(ph[0][mask].mean() / rec[0][mask].mean())


@torch.no_grad()
def insert_lowdose_noise(hu_slices: torch.Tensor, pixel_mm: float, dose_fraction: float, ct: ParallelBeamCT,
                         i0: float = 2.0e5, sigma_e: float = 10.0, window: str = "shepp-logan",
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Simulate a reduced-dose acquisition of full axial slices.

    hu_slices: (B,N,N) HU images (full field of view, N == ct.n)
    pixel_mm : in-plane pixel spacing in mm
    returns  : (B,N,N) HU images at the requested dose fraction
    """
    if dose_fraction >= 0.999:
        return hu_slices.clone()
    ep = attenuation_factor(hu_slices, pixel_mm, ct)
    return hu_slices + noise_from_ep(ep, pixel_mm, dose_fraction, ct, i0, sigma_e, window, generator).to(hu_slices.dtype)


def attenuation_factor(hu_slices: torch.Tensor, pixel_mm: float, ct: ParallelBeamCT) -> torch.Tensor:
    """exp(line integral) for every ray, (B,A,M). Depends only on the anatomy, so it is computed
    ONCE per slice and re-used for every simulated dose / kernel (about 2x faster bank building)."""
    mu = hu_to_mu(hu_slices.to(ct.dtype))
    p = ct.forward_project(mu) * pixel_mm
    return torch.exp(torch.clamp(p, max=12.0))


def noise_from_ep(ep: torch.Tensor, pixel_mm: float, dose_fraction: float, ct: ParallelBeamCT, i0=2.0e5,
                  sigma_e: float = 10.0, window: str = "shepp-logan", generator=None) -> torch.Tensor:
    """Added-noise image (HU) for dose fraction a. i0: float, or tensor (B,) = per-slice incident flux."""
    a = float(dose_fraction)
    if a >= 0.999:
        return torch.zeros(ep.shape[0], ct.n, ct.n, device=ep.device, dtype=ep.dtype)
    if torch.is_tensor(i0):
        i0 = i0.to(ep.device, ep.dtype).view(-1, 1, 1)
    var = (1 - a) / a * ep / i0 * (1 + (1 + a) / a * sigma_e ** 2 * ep / i0)
    # photon starvation: a detector never reads fewer than ~1 photon, so the noise of a measured line
    # integral is bounded (std ~1). Without this cap, rays through metal / very dense paths at 5 % dose
    # explode to absurd values (float overflow).
    var = torch.clamp(var, max=1.0)
    z = torch.randn(ep.shape, device=ep.device, dtype=ep.dtype, generator=generator)
    noise_p = torch.sqrt(var) * z / pixel_mm  # back to "pixel-sum" units expected by FBP
    return 1000.0 * ct.fbp(noise_p, window) / MU_WATER


def kernel_proxy(hu_slices: torch.Tensor, kind: str) -> torch.Tensor:
    """Image-domain proxy of a reconstruction-kernel change (applied after noise insertion).
    smooth: Gaussian blur (sigma 0.7 px); sharp: unsharp masking (amount 0.8, sigma 1 px)."""
    if kind in (None, "standard", "none"):
        return hu_slices

    def gblur(x, s):
        r = int(math.ceil(3 * s))
        t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum()
        x4 = x.unsqueeze(1)
        x4 = F.conv2d(F.pad(x4, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        x4 = F.conv2d(F.pad(x4, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
        return x4[:, 0]

    if kind == "smooth":
        return gblur(hu_slices, 0.7)
    if kind == "sharp":
        return hu_slices + 0.8 * (hu_slices - gblur(hu_slices, 1.0))
    raise ValueError(kind)


# Noise texture of the 'standard' acquisition: un-apodised ramp (Ram-Lak). With the bilinear projector this
# reproduced the noise power spectrum of the real Mayo 10%-dose chest data (B50f kernel) much better than
# Shepp-Logan (NPS peak 0.28 vs 0.34 cycles/pixel real; Shepp-Logan peaked at 0.19). See scripts/06.
KERNEL_WINDOW = {"standard": "ramlak", "smooth": "hann", "sharp": "ramlak"}


def slice_i0(mas, thickness_mm, cfg) -> "np.ndarray":
    """Effective incident photons per ray for each slice.
    i0_mode 'per_mAs' (default): I0 = i0_per_mAs * mAs * (thickness / 1.5 mm), i.e. quantum noise variance
    ~ 1 / (mAs x slice thickness). Calibrated on Mayo chest data (1.5 mm) by scripts/06_validate_noise_sim.py.
    Slices without a usable mAs fall back to the scan median, then to the constant cfg['i0']."""
    import numpy as np
    mas = np.asarray(mas, dtype=np.float64)
    if cfg.get("i0_mode", "constant") != "per_mAs":
        return np.full(mas.shape, float(cfg["i0"]))
    good = np.isfinite(mas) & (mas > 0)
    if not good.any():
        return np.full(mas.shape, float(cfg["i0"]))
    mas = np.where(good, mas, np.median(mas[good]))
    thk = float(thickness_mm) if thickness_mm and np.isfinite(thickness_mm) and thickness_mm > 0 else 1.5
    thk = min(max(thk, 0.5), 5.0)
    return float(cfg["i0_per_mAs"]) * np.clip(mas, 5.0, 1000.0) * thk / 1.5
