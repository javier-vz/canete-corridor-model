# Reproduction guide

This document tells you exactly how to reproduce each numerical result and figure of the paper. Follow the steps in order.

**Estimated total wall time for the full paper reproduction:** ~40 hours on a 7-core laptop. Most of the time is in the (ε, κ) phase scan (30 h); everything else can be done in an evening.

---

## Prerequisites

- Python 3.10 or newer
- Approximately 500 MB of free disk space for intermediate results
- Recommended: at least 8 GB of RAM

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 0 — Build the state cache

Everything downstream depends on the state object, which is built once from the CSV and cached to disk. This takes ~30 seconds.

```bash
python src/build_state.py --csv data/grilla_con_W_hidrica.csv --out state_cache.pkl
```

Expected output:
```
Loaded 12826 valid cells from data/grilla_con_W_hidrica.csv
Built graph: 12826 nodes, 50624 edges
Extracted 24 attractors at quantile 0.92
Selected 86 attractor pairs (6-nearest-neighbours, deduplicated)
Wrote state_cache.pkl (7.5 MB)
```

---

## Step 1 — Quick sanity check (2 min)

Run one CA configuration to verify everything works end to end:

```bash
python src/run_single.py --state state_cache.pkl --eps 0.15 --kappa 0.03 --seed 42
```

Expected output near:
```
AUC       = 0.816
LCC size  = 138
m1        = 0.824
m2        = 0.348
m3        = 0.003
runtime   ~ 7 min single-CPU
```

If your numbers differ by more than the third decimal, check that your NumPy and SciPy versions match `requirements.txt`.

---

## Step 2 — Figures

### Figure 2 (input surfaces)

```bash
python scripts/regen_figure2.py
```

Produces `figure2_corrected.png`. Copy it to your paper directory as `v47_inputs_surfaces.png`. Wall time: seconds.

### Figures 3, 4, 5 (with scalebar + north arrow)

```bash
python scripts/regen_figures_with_scalebar.py
```

Produces `v47_E.png` (Fig. 3), `figure4_twopanel.png` (Fig. 4, rename to `v47_traffic.png` in your paper), and `vz1_zones_L_mean.png` (Fig. 5). Wall time: ~25 min (Fig. 3 is instantaneous; Fig. 4 requires two CA runs of ~7 min; Fig. 5 requires one).

---

## Step 3 — Numerical results

### KS diagnostic (§4.3)

```bash
python src/analyze/analyze_ks_test.py --state state_cache.pkl
```

Produces `ks_report.txt`. The paper reports:
- Full sites, L: D = 0.543, p = 1.87e-23
- Leave-dominant-out, L: D = 0.291, p = 0.054
- Leave-dominant-out, E: D = 0.432, p = 6.65e-04

Wall time: seconds.

### FETE baseline (§4.5, Table 3)

Two-step: run experiments then aggregate.

```bash
# 1) Run experiments (~1 hour on 7 workers)
python scripts/run_fete_local.py

# 2) Aggregate
python src/analyze/analyze_fete.py
```

Produces `results_fete/` (150 JSONs, 5 configurations × 30 seeds) and `fete_summary.txt`. The paper reports at n_pairs = 86:
- AUC = 0.810 ± 0.008
- LCC = 146 ± 20
- m2 = 0.27 ± 0.05

---

## Step 4 — Sensitivity analyses

### Weight ablation (§3.6)

```bash
python scripts/run_ablation_weights.py
```

Produces `results_ablation_weights/` (51 JSONs, 17 configurations × 3 seeds), `ablation_weights_summary.csv`, and `ablation_weights_summary.txt`. Wall time: ~6 hours on 7 workers.

The paper reports:
- AUC ∈ [0.811, 0.822], range 0.011
- Zonal ρ ∈ [0.544, 0.548], range 0.004

### (ε, κ) phase scan (§3.6)

```bash
# The big one — ~30 hours on 7 workers.
python scripts/run_phase1_local.py

# Then aggregate:
python src/analyze/analyze_phase1.py
```

Produces `results_phase1/` (1210 JSONs, 121 grid points × 10 seeds), `phase1_aggregate.csv`, `phase1_overview.png`, `phase1_susceptibilities.png`, `phase1_ridge_detection.png`, and `phase1_summary.txt`. The paper reports:
- AUC ∈ [0.814, 0.818] across the entire scan

You can preview partial results at any time with:
```bash
python scripts/preview_phase1.py
```

---

## Interruption and resumption

Every long-running script is checkpointed at the granularity of individual jobs: each (ε, κ, seed) or (n_pairs, seed) or (config, seed) writes a separate JSON on completion. If a script is interrupted, re-running it will skip already-completed jobs and continue from where it left off. Safe to Ctrl-C at any time.

---

## Notes on numerical reproducibility

The kernel is deterministic given fixed `numpy` random seeds. However:

- The order in which jobs complete in a parallel run can vary, which does **not** affect any per-run numbers
- Different `numpy`/`scipy` versions may produce differences at the 4th-5th decimal
- Small differences between the version reported in the paper and yours (order of 1e-3 in AUC) are within acceptable numerical noise
- The K-means partition (used for zonal metrics) is fixed via `random_state=0` in `sklearn.cluster.KMeans`, so zonal Spearman correlations should be reproducible to full precision

If you observe large deviations, please open a GitHub issue with your Python and library versions.
