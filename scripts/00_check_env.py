"""Step 0 - check Python packages, GPU and simulator speed.
Run:  python scripts/00_check_env.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dart.ct_physics import ParallelBeamCT, insert_lowdose_noise

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU    : {p.name}  |  VRAM {p.total_memory / 1e9:.1f} GB")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for mod in ["pydicom", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "openpyxl"]:
    try:
        __import__(mod); print(f"  ok  {mod}")
    except ImportError:
        print(f"  MISSING {mod}  -> pip install -r requirements.txt")

n = 512
ct = ParallelBeamCT(n, 360, device=dev, angle_chunk=30)
x = torch.full((8, n, n), -1000.0, device=dev)
yy, xx = torch.meshgrid(torch.arange(n, device=dev) - n / 2, torch.arange(n, device=dev) - n / 2, indexing="ij")
x[:, ((xx * 0.7 / 150) ** 2 + (yy * 0.7 / 110) ** 2) < 1] = 40.0
insert_lowdose_noise(x[:1], 0.7, 0.25, ct)  # warm-up
if dev.type == "cuda":
    torch.cuda.synchronize()
t = time.time()
y = insert_lowdose_noise(x, 0.7, 0.10, ct)
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t) / 8
print(f"Low-dose simulation: {dt * 1000:.0f} ms per 512x512 slice on {dev}")
print(f"Added noise at 10% dose in a water ellipse: {(y - x)[0][x[0] > 0].std().item():.1f} HU")
print("Environment OK." if dev.type == "cuda" else "WARNING: running on CPU - training will be very slow.")
