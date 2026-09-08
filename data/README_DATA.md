# Data documentation

This directory contains the processed data required to reproduce the paper.

---

## Files

### `DEM_vallebajo.tif` (6.6 MB)

**Description:** Digital elevation model of the lower Cañete Valley study area, resampled from SRTM 1-arc-second (~30 m) global DEM.

**Source:** NASA Shuttle Radar Topography Mission Version 3 (SRTM V3). Original tile downloaded via the USGS EarthExplorer service.

**Projection:** UTM Zone 18S (EPSG:32718)

**Extent:** approximately 76.4°W to 76.0°W (west to east); 13.1°S to 12.7°S (south to north). The tile has been clipped to the study area used throughout the paper.

**License:** SRTM data are in the public domain (US Government work).

**Usage in the pipeline:** The DEM is the source of the `altitud` (elevation) and `pendiente` (slope, computed from the 3×3 spatial derivative) columns in `grilla_con_W_hidrica.csv`. It is included in the repository for two reasons:

1. To enable full reproducibility of the CSV construction pipeline (see `notebooks/02_data_preparation.ipynb`)
2. To allow future users to compute variables at different resolutions or with different algorithms.

For the paper analysis itself, only `grilla_con_W_hidrica.csv` is needed.

---

### `grilla_con_W_hidrica.csv` (2.4 MB)

**Description:** The 250 m analysis grid used throughout the paper. Each row is one cell of the study area (12,826 valid cells total).

**Columns:**

| Column | Type | Description |
|---|---|---|
| `X`, `Y` | float | UTM 18S coordinates of the cell centre, in metres |
| `altitud` | float | Elevation above sea level in metres, averaged from the SRTM DEM within the cell |
| `pendiente` | float | Slope in degrees, computed from the 3×3 spatial derivative on the aggregated DEM |
| `W_river` | float in [0, 1] | Inverse normalised distance to the nearest river line: 1 - dist_river / max(dist_river). Rivers are taken from HydroRIVERS v1.0 (Lehner & Grill 2013) |
| `dist_river_m` | float | Raw Euclidean distance from cell centre to the nearest river line, in metres |
| `W_corridor` | float in [0, 1] | Hydrological corridor variable: W_river × (1 - pendiente_norm). This is the input $W$ used to build $E$ and $c$ in the paper |
| `presencia_sitio_nuevo` | int (0 or 1) | Binary flag indicating whether the cell intersects a MINCUL-registered archaeological site or a site verified during the 2022 field campaigns (Cornejo Meza 2022) |

**Provenance of the archaeological sites:**

- Ministerio de Cultura del Perú (MINCUL) heritage registry, obtained via the SIGDA system
- Field-verified sites from the 2022 prospection campaign led by Fernandini (see Cornejo Meza 2022 undergraduate thesis, PUCP)

Coordinates of individual sites are aggregated to the 250 m grid; the CSV does not contain per-site attributes.

**License:** The CSV as a whole is released under CC-BY-4.0 by the authors of this repository (attribution: Vera Zúñiga & Cornejo Meza 2026). Note that the underlying MINCUL archaeological data are subject to Peruvian law regarding cultural patrimony; use of derived location information for non-scientific purposes may require authorisation from the Ministerio de Cultura.

**How this CSV was built:** See `notebooks/02_data_preparation.ipynb` and the original script `original_scripts/CA1.py`. In summary:

1. The SRTM DEM was resampled to a 250 m grid over the study extent, computing per-cell mean elevation
2. Slope was computed from the 3×3 spatial derivative of the aggregated DEM
3. HydroRIVERS v1.0 was clipped to a 5 km buffer around the study grid, and for each cell the Euclidean distance to the nearest river line was computed via GeoPandas
4. The archaeological presence flag was set by intersecting each cell with the MINCUL point layer and the 2022 field-verified point layer

The min–max normalisations required by the paper's equations (h̃, p̃, W, and the derived E, c, T) are applied at load time by `src/build_state.py`, not stored in the CSV.
