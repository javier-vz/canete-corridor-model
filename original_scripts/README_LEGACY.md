# Original (legacy) scripts

These are the original scripts used during the development of the paper, preserved here for transparency and archival reference. They are **not** the recommended entry point for reproducing the paper — use `src/` and `scripts/` for that.

---

## Files

### `CA1.py`

The script that built `data/grilla_con_W_hidrica.csv` from the SRTM DEM and the HydroRIVERS shapefile. Key operations:

1. Reads the DEM as a raster
2. Loads HydroRIVERS v1.0 river lines, clipped to a buffer around the study area
3. Constructs a 250 m grid over the study extent
4. For each cell, computes:
   - Mean elevation (`altitud`)
   - Slope in degrees (`pendiente`)
   - Distance to the nearest river line (`dist_river_m`)
   - `W_river = 1 - dist_river_norm`
   - `W_corridor = W_river * (1 - pendiente_norm)`
5. Writes the resulting table as CSV

**Note:** This script was used once to produce the CSV shipped in `data/`. The equivalent modern pipeline is documented in `notebooks/02_data_preparation.ipynb`.

### `CA4.7.py`

The monolithic pipeline script used to produce the results reported in the first submission. It combines: loading the CSV, computing E and c, extracting attractors, running the CA with perceptual noise and congestion, aggregating traffic, and producing summary plots.

The refactored kernel in `src/ca_kernel.py` is functionally equivalent (verified: same AUC to 4 decimal places at the canonical configuration) but split into reusable functions and packaged for use in parallel experiments (Phase 1 scan, FETE baseline, weight ablation). For a mapping between `CA4.7.py`'s logic and the refactored kernel, see `notebooks/03_original_pipeline.ipynb`.

---

## Why keep them?

Reviewers and future researchers should be able to inspect the exact pre-refactor code that produced the numbers we first reported. The refactored version in `src/` is what we recommend using going forward, but the originals are the ground truth of provenance.
