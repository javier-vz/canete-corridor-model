# Architecture

This document explains how the modules in this repository fit together and why the code is structured the way it is.

---

## The three layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: experiments                                        │
│  scripts/                                                    │
│  - run_phase1_local.py       exp1_phase_scan.py              │
│  - run_fete_local.py         exp0_fete_baseline.py           │
│  - run_ablation_weights.py                                   │
│  Loop over parameter grids, multiprocessing, checkpoint      │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: kernel + build                                     │
│  src/                                                        │
│  - ca_kernel.py       core CA + FETE                         │
│  - build_state.py     builds the immutable state             │
│  - run_single.py      one job, one config, one seed          │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: data                                               │
│  data/                                                       │
│  - grilla_con_W_hidrica.csv                                  │
│  - DEM_vallebajo.tif                                         │
└──────────────────────────────────────────────────────────────┘
```

**Post-processing layer** (`src/analyze/`) reads the JSON output of Layer 3 experiments and produces summary tables and plots.

---

## Why this split?

The original codebase (`original_scripts/CA4.7.py`) was a monolithic script that loaded the CSV, computed everything, ran one CA configuration, and produced plots — all in one pass. This was fine for producing the first submission but had two limits:

1. **Cannot parallelise:** to run 1000+ configurations with different seeds and parameters, you cannot re-load the CSV and rebuild the graph every time. The refactor separates the expensive one-time build (`build_state.py`) from the cheap per-run inner loop (`ca_kernel.run_one`).
2. **Cannot iterate on plots without recomputing:** by writing intermediate results to JSON, plots and analyses can be regenerated in seconds without re-running the CA.

The refactor is functionally equivalent to `CA4.7.py`: same AUC to 4 decimal places at the canonical configuration `(ε=0.15, κ=0.03, seed=42)`.

---

## Data flow

For a single CA run:

```
data/grilla_con_W_hidrica.csv
        │
        ▼ build_state.py
state_cache.pkl  (contains: h, p, W, T grids; E, c surfaces;
                  8-neighbour graph with base weights; 24 attractors;
                  86 attractor pairs; site mask)
        │
        ▼ ca_kernel.run_one(state, eps, kappa, seed)
result JSON  (contains: AUC, LCC size, m1, m2, m3,
              F_norm_sum_check, runtime, host)
        │
        ▼ analyze/*.py
summary tables and plots
```

For parallel experiments (Phase 1, FETE, weight ablation):

```
scripts/run_*_local.py
        │
        ├── multiprocessing.Pool with N workers
        │       │
        │       ▼ worker: loads state once, then runs many jobs
        │       │
        │       ▼ ca_kernel.run_one (or run_fete)
        │       │
        │       ▼ writes one JSON per job to results_*/
        │
        ▼ analyze/*.py reads all JSONs in the directory
```

---

## Key design decisions

**State is immutable.** Once `state_cache.pkl` is built, no downstream code modifies it. The base graph in the state is the reference; per-run perturbations create a `deepcopy` inside `run_one`. This makes the code safe to run in parallel: no worker can corrupt another's state.

**Every job is atomic.** Each `(eps, kappa, seed)` writes exactly one JSON. If a run is interrupted mid-experiment, the next execution skips completed JSONs and continues. No aggregation happens during the experiment; it is always deferred to a separate `analyze_*.py` step.

**No global randomness.** All randomness in the kernel is controlled by an explicit `numpy.random.Generator` seeded per job. Reproducibility is total: same seed, same result.

**Kernel exports both CA and FETE.** `ca_kernel.py` provides `run_one()` (CA canonical) and `run_fete()` (baseline). Both share the same fixation field construction, evaluation code, and grid representation, so the comparison between them (Table 3 in the paper) uses identical downstream machinery. Only the routing engine differs.

---

## When to use which script

| I want to... | Use |
|---|---|
| Run one CA configuration to inspect | `src/run_single.py` |
| Reproduce Table 3 (FETE baseline) | `scripts/run_fete_local.py` + `src/analyze/analyze_fete.py` |
| Reproduce §3.6 sensitivity (weights) | `scripts/run_ablation_weights.py` |
| Reproduce §3.6 sensitivity (ε, κ) | `scripts/run_phase1_local.py` + `src/analyze/analyze_phase1.py` |
| Reproduce §4.3 (KS diagnostics) | `src/analyze/analyze_ks_test.py --state state_cache.pkl` |
| Regenerate a figure | `scripts/regen_figure2.py` or `scripts/regen_figures_with_scalebar.py` |
| See how the pipeline works interactively | `notebooks/01_reproduce_paper_results.ipynb` |
| Understand the original monolithic script | `notebooks/03_original_pipeline.ipynb` |
| Rebuild the CSV from scratch | `notebooks/02_data_preparation.ipynb` |
