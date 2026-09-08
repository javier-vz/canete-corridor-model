# canete-corridor-model

Reproducibility repository for **"An Unsupervised Corridor-Based Model for Archaeological Prospection in the Lower Cañete Valley, Peru"** (Vera Zúñiga & Cornejo Meza, submitted to *Journal of Computer Applications in Archaeology*).

This repository provides the code, processed data, and computational recipes needed to reproduce every quantitative result and figure in the paper.

---

## What is in this repository

The model treats archaeological prospection as an unsupervised movement-corridor problem. Starting from a 250 m analysis grid over the lower Cañete Valley, it derives an environmental affinity field $E$ and a cost surface $c$ from four normalised inputs (elevation, slope, hydrological corridor, terrace suitability), extracts a set of 24 attractor cells from $E$, and computes an ensemble of $k$-shortest paths between attractor pairs on the cost graph. The accumulated traffic field $F$ and its blend $L$ with the terrace layer are then evaluated against 84 archaeological sites at cell scale (AUC, buffered overlap, Kolmogorov–Smirnov) and at zonal scale (Spearman correlation over K-means partitions).

The repo covers **only the JCAA paper** (Cañete). A separate physics-oriented analysis of the phase transition observed in the traffic field is planned as a companion Letter and is not included here.

## Repository layout

```
canete-corridor-model/
├── README.md
├── LICENSE                  MIT
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── DEM_vallebajo.tif                SRTM 1-arc-second DEM, clipped to the study area
│   ├── grilla_con_W_hidrica.csv         Precomputed 250-m grid with all input variables
│   └── README_DATA.md                   Provenance, columns, licence
├── src/
│   ├── ca_kernel.py                     Core CA + FETE implementation
│   ├── build_state.py                   Builds the state object from the CSV
│   ├── run_single.py                    Single-job runner (cluster-friendly)
│   └── analyze/
│       ├── analyze_phase1.py            Post-processes the (eps, kappa) scan
│       ├── analyze_fete.py              Aggregates FETE runs
│       └── analyze_ks_test.py           Computes KS diagnostics
├── scripts/
│   ├── run_phase1_local.py              Phase-1 scan, multiprocessing (~30 h)
│   ├── run_fete_local.py                FETE baseline, multiprocessing (~1 h)
│   ├── run_ablation_weights.py          Weight ablation (~6 h)
│   ├── preview_phase1.py                Quick sanity check
│   ├── regen_figure2.py                 Regenerates Figure 2 with scalebar
│   └── regen_figures_with_scalebar.py   Regenerates Figures 3, 4, 5
├── notebooks/
│   ├── 01_reproduce_paper_results.ipynb Walkthrough of every quantitative result
│   ├── 02_data_preparation.ipynb        How the CSV was built from DEM + HydroRIVERS
│   └── 03_original_pipeline.ipynb       CA4.7 explained
├── original_scripts/
│   ├── CA1.py                           Original grid construction (Vera & Cornejo Meza)
│   ├── CA4.7.py                         Original monolithic pipeline (Vera & Cornejo Meza)
│   └── README_LEGACY.md
└── docs/
    ├── REPRODUCE.md                     Step-by-step reproduction of every paper figure
    ├── ARCHITECTURE.md                  How the modules fit together
    └── FAQ.md
```

## Quick start (30 seconds)

```bash
git clone https://github.com/YOUR-USERNAME/canete-corridor-model.git
cd canete-corridor-model
pip install -r requirements.txt

# Build the state (parses CSV, computes E and c, extracts attractors, etc.)
python src/build_state.py --csv data/grilla_con_W_hidrica.csv --out state_cache.pkl

# Run one CA configuration
python src/run_single.py --state state_cache.pkl --eps 0.15 --kappa 0.03 --seed 42
```

You should see something close to:
```
AUC       = 0.816
LCC size  = 138
m2        = 0.348
runtime   = ~7 min (single-CPU laptop; scales linearly with attractor pairs)
```

## Reproducing the paper

For a step-by-step guide to reproducing each numerical result and figure, see **`docs/REPRODUCE.md`**. In short:

| Paper output | Command | Wall time |
|---|---|---|
| Fig. 2 (input surfaces) | `python scripts/regen_figure2.py` | seconds |
| Figs. 3, 4, 5 (with scalebar) | `python scripts/regen_figures_with_scalebar.py` | ~25 min |
| FETE baseline (Table 3, §4.5) | `python scripts/run_fete_local.py && python src/analyze/analyze_fete.py` | ~1 h |
| KS diagnostic (§4.3) | `python src/analyze/analyze_ks_test.py --state state_cache.pkl` | seconds |
| Weight ablation (§3.6) | `python scripts/run_ablation_weights.py` | ~6 h |
| (Eps, kappa) phase scan (§3.6) | `python scripts/run_phase1_local.py` | ~30 h |

Runtimes are measured on a Ryzen 7 5825U (8 physical cores, 15 GB RAM), Windows 11 + Anaconda `soft` environment. Wall times scale roughly with the number of physical cores available.

## Data

The main dataset (`data/grilla_con_W_hidrica.csv`) is a 12,826-row table with one row per 250 m cell of the analysis grid. Columns include the four normalised inputs, the derived surfaces $E$, $c$, $T$, and the binary flag indicating whether the cell intersects a MINCUL-registered archaeological site.

See `data/README_DATA.md` for a full data dictionary, provenance of the elevation, hydrological, and archaeological sources, and licensing notes.

## How to cite

If you use this software or the derived data, please cite the accompanying paper (see `CITATION.cff`). Bibliographic details will be added once the paper is accepted.

## License

MIT. See `LICENSE`.

## Contact

- Javier Vera Zúñiga: see repository maintainer profile
- Dina Soledad Cornejo Meza: Pontificia Universidad Católica del Perú

For questions about the archaeological data or its interpretation, contact the authors. For issues with the code, open a GitHub issue.
