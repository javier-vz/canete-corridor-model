# canete-corridor-model

Reproducibility repository for **"An Unsupervised Corridor-Based Model for Archaeological Prospection in the Lower Cañete Valley, Peru"** (Cornejo Meza & Vera Zúñiga, submitted to *Journal of Computer Applications in Archaeology*).

This repository provides the code, processed data, and computational recipes needed to reproduce every quantitative result and figure in the paper.

---

## What is in this repository

The model treats archaeological prospection as an unsupervised movement-corridor problem. Starting from a 250 m analysis grid over the lower Cañete Valley, it derives an environmental affinity field $E$ and a cost surface $c$ from four normalised inputs (elevation, slope, hydrological corridor, terrace suitability), extracts a set of 24 attractor cells from $E$, and computes an ensemble of $k$-shortest paths between attractor pairs on the cost graph. The accumulated traffic field $F$ and its blend $L$ with the terrace layer are then evaluated against 84 archaeological sites at cell scale (AUC, buffered overlap, Kolmogorov–Smirnov) and at zonal scale (Spearman correlation over K-means partitions).

The repo covers **only the JCAA paper** (Cañete).

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
│   ├── run_fete_1mE.py                  FETE with 1-E cost surface, matched seeds (~1 h)
│   ├── rerun_ablations_seed42.py        Weight / T-parameter / quantile ablations, K-means seed 42 (~40 min)
│   ├── preview_phase1.py                Quick sanity check
│   ├── regen_figure1_studyarea.py       Regenerates Fig. 1 (study area map)
│   ├── regen_figure2.py                 Regenerates Fig. 3 (input surfaces)
│   ├── regen_figure_E.py                Regenerates Fig. 4 (E field with attractors)
│   ├── regen_figure4.py                 Regenerates Fig. 5 (traffic F, baseline vs canonical)
│   └── regen_figures_zonal.py           Regenerates Figs. 6, 7, 8 (zonal analysis)
├── notebooks/
│   ├── 01_reproduce_paper_results.ipynb Walkthrough of every quantitative result
│   ├── 02_data_preparation.ipynb        How the CSV was built from DEM + HydroRIVERS
│   └── 03_original_pipeline.ipynb       CA4.7 explained
├── original_scripts/
│   ├── CA1.py                           Original grid construction (Cornejo Meza & Vera Zúñiga)
│   ├── CA4.7.py                         Original monolithic pipeline (Cornejo Meza & Vera Zúñiga)
│   └── README_LEGACY.md
└── docs/
    ├── REPRODUCE.md                     Step-by-step reproduction of every paper figure
    ├── ARCHITECTURE.md                  How the modules fit together
    └── FAQ.md
```

## Quick start (30 seconds)

```bash
git clone https://github.com/javier-vz/canete-corridor-model.git
cd canete-corridor-model
pip install -r requirements.txt

# Build the state (parses CSV, computes E and c, extracts attractors, etc.)
python src/build_state.py --csv data/grilla_con_W_hidrica.csv --out state_cache.pkl

# Run one CA configuration (note the required `ca` subcommand and `--out`)
python src/run_single.py ca \
    --eps 0.15 --kappa 0.03 --seed 42 \
    --state state_cache.pkl \
    --out results/run_e0.15_k0.03_s42.json
```

You should see output similar to:

```
[done] results/run_e0.15_k0.03_s42.json  (~7 min)  m1=... m2=... AUC=0.81x
```

The exact values will differ across seeds. The Table 3 values reported in the paper (AUC = 0.816 ± 0.002, LCC = 138 ± 23, m₂ = 0.348 ± 0.061) are the **mean and standard deviation over 30 seeds** at the canonical configuration, not the output of any single run. See `docs/REPRODUCE.md` for how to reproduce those aggregated numbers.

## Reproducing the paper

For a step-by-step guide to reproducing each numerical result and figure, see **`docs/REPRODUCE.md`**. In short:

| Paper output | Command | Wall time |
|---|---|---|
| Fig. 1 (study area) | `python scripts/regen_figure1_studyarea.py` | seconds |
| Fig. 3 (input surfaces) | `python scripts/regen_figure2.py --csv data/grilla_con_W_hidrica.csv --out figure3.png` | seconds |
| Fig. 4 (E with attractors) | `python scripts/regen_figure_E.py --csv data/grilla_con_W_hidrica.csv --out figure4.png` | seconds |
| Fig. 5 (traffic F, baseline vs canonical) | `python scripts/regen_figure4.py` | ~4 min |
| Figs. 6, 7, 8 (zonal analysis) | `python scripts/regen_figures_zonal.py --csv data/grilla_con_W_hidrica.csv` | ~2 min |
| FETE baseline (Table 3, §4.5) | `python scripts/run_fete_local.py && python src/analyze/analyze_fete.py` | ~1 h |
| FETE 1−E variant, matched seeds (Table 3, §4.5) | `python scripts/run_fete_1mE.py` | ~1 h |
| KS diagnostic (§4.3) | `python src/analyze/analyze_ks_test.py --state state_cache.pkl` | seconds |
| Weight / T / quantile ablations (§3.6) | `python scripts/rerun_ablations_seed42.py` | ~40 min |
| (ε, κ) phase scan (§3.6) | `python scripts/run_phase1_local.py` | ~30 h |

Runtimes are measured on a Ryzen 7 5825U (8 physical cores, 15 GB RAM), Windows 11 + Anaconda `soft` environment. Wall times scale roughly with the number of physical cores available.

## Data

The main dataset (`data/grilla_con_W_hidrica.csv`) is a 12,826-row table with one row per 250 m cell of the analysis grid. Columns include the four normalised inputs, the derived surfaces $E$, $c$, $T$, and the binary flag indicating whether the cell intersects a MINCUL-registered archaeological site.

See `data/README_DATA.md` for a full data dictionary, provenance of the elevation, hydrological, and archaeological sources, and licensing notes.

## K-means partition seed

All zonal correlation results reported in the paper (Fig. 5 zonal map, Table 2 ρ values, Fig. 8 correlation bars, and the ρ values in the sensitivity ablations of §3.6) use `KMeans(random_state=42, n_init=10)`. The `rerun_ablations_seed42.py` script uses this same seed to keep every ρ directly comparable to the ρ = 0.638 baseline of the canonical configuration.

## How to cite

If you use this software or the derived data, please cite the accompanying paper (see `CITATION.cff`). Bibliographic details will be updated once the paper is accepted. A DOI-versioned snapshot of this repository will be deposited in Zenodo upon acceptance, and the DOI will be added to `CITATION.cff` at that time.

## License

MIT. See `LICENSE`.

## Contact

- Soledad Cornejo Meza: Pontificia Universidad Católica del Perú
- Javier Vera Zúñiga (corresponding): jveraz@utp.edu.pe, Universidad Tecnológica del Perú

For questions about the archaeological data or its interpretation, contact the authors. For issues with the code, open a GitHub issue.
