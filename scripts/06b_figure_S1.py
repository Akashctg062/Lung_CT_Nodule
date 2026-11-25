"""Re-draw the composite supplementary Fig. S1 (800 dpi) from the data step 06 saved; no GPU or torch needed.
Run: python scripts/06b_figure_S1.py --config configs/default.yaml
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from dart.figs1 import make_figS1

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
with open(ap.parse_args().config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
out, rep = make_figS1(os.path.join(cfg["work_dir"], "results"))
print(f"composite Fig. S1 (representative case {rep}) -> {out}")
