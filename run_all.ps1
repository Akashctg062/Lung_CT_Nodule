# Run the complete DART-Nodule study (PowerShell, from the repository folder, virtual environment activated)
# Usage:  .\run_all.ps1
$ErrorActionPreference = "Stop"
$cfg = "configs/default.yaml"
python scripts/00_check_env.py
python scripts/01_index_lidc.py         --config $cfg
python scripts/06_validate_noise_sim.py --config $cfg   # needs the Mayo chest cases; comment out if not downloaded
python scripts/06b_figure_S1.py         --config $cfg   # composite simulator-validation figure
python scripts/02_build_bank.py         --config $cfg
python scripts/03_train.py              --config $cfg
python scripts/04_evaluate.py           --config $cfg
python scripts/05_lungx_external.py     --config $cfg   # needs LUNGx; comment out if not downloaded
python scripts/07_example_figure.py     --config $cfg
python scripts/08_conformal_figures.py  --config $cfg
python scripts/09_extra_analyses.py     --config $cfg
Write-Host "All done. Results are in work/results."
