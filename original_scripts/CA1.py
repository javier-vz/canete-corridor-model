# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box

BASE_DIR = Path(".")   # carpeta actual: CA-arch

# buscar archivos automáticamente
grid_candidates = list(BASE_DIR.rglob("grilla_final_modelo_logistico.csv"))
if not grid_candidates:
    raise FileNotFoundError("No encontré grilla_final_modelo_logistico.csv")
GRID_CSV = grid_candidates[0]

shp_candidates = list(BASE_DIR.rglob("*.shp"))
hydro_candidates = [p for p in shp_candidates if "hydro" in str(p).lower()]
if not hydro_candidates:
    raise FileNotFoundError("No encontré el shapefile de HydroRIVERS")
HYDRO_SHP = hydro_candidates[0]

print("Usando grilla:", GRID_CSV)
print("Usando río:", HYDRO_SHP)

# leer grilla
df = pd.read_csv(GRID_CSV)

required_cols = ["X", "Y", "altitud", "pendiente"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas: {missing}")

gdf_grid = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["X"], df["Y"]),
    crs="EPSG:32718"
)

# leer ríos
gdf_rivers = gpd.read_file(HYDRO_SHP)
if gdf_rivers.crs != gdf_grid.crs:
    gdf_rivers = gdf_rivers.to_crs(gdf_grid.crs)

# recortar al entorno de la grilla
buffer_m = 5000
xmin, ymin, xmax, ymax = gdf_grid.total_bounds
bbox = gpd.GeoDataFrame(
    geometry=[box(xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)],
    crs=gdf_grid.crs
)

gdf_rivers_clip = gpd.clip(gdf_rivers, bbox)
if len(gdf_rivers_clip) == 0:
    raise ValueError("No quedó ningún río tras el recorte")

river_union = gdf_rivers_clip.geometry.union_all()

# distancia al río
gdf_grid["dist_river_m"] = gdf_grid.geometry.distance(river_union)

def minmax(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

gdf_grid["dist_river_norm"] = minmax(gdf_grid["dist_river_m"])
gdf_grid["p_norm"] = minmax(gdf_grid["pendiente"])
gdf_grid["h_norm"] = minmax(gdf_grid["altitud"])

# capas hídricas
gdf_grid["W_river"] = 1.0 - gdf_grid["dist_river_norm"]
gdf_grid["W_corridor"] = gdf_grid["W_river"] * (1.0 - gdf_grid["p_norm"])

# guardar csv nuevo
out_csv = BASE_DIR / "grilla_con_W_hidrica.csv"
pd.DataFrame(gdf_grid.drop(columns="geometry")).to_csv(out_csv, index=False, encoding="utf-8")
print("Guardado:", out_csv)

# figura rápida
xs = np.sort(gdf_grid["X"].unique())
ys = np.sort(gdf_grid["Y"].unique())
x_to_ix = {x: i for i, x in enumerate(xs)}
y_to_iy = {y: i for i, y in enumerate(ys)}

ny, nx = len(ys), len(xs)
W_river_grid = np.full((ny, nx), np.nan)
W_corridor_grid = np.full((ny, nx), np.nan)

for _, row in gdf_grid.iterrows():
    iy = y_to_iy[row["Y"]]
    ix = x_to_ix[row["X"]]
    W_river_grid[iy, ix] = row["W_river"]
    W_corridor_grid[iy, ix] = row["W_corridor"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

im1 = axes[0].imshow(W_river_grid, origin="lower", interpolation="nearest")
axes[0].set_title("W_river")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(W_corridor_grid, origin="lower", interpolation="nearest")
axes[1].set_title("W_corridor")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("W_hidrica_diagnostico.png", dpi=300, bbox_inches="tight")
plt.show()