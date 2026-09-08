# -*- coding: utf-8 -*-
"""
CA4.6.py
V4.6 — ajustes D y E sobre V4.5

Cambios puntuales respecto a V4.5:

D) Ruido perceptual en el costo (stochastic cost perception)
   - Para cada par (a,b), los costos de aristas se perturban:
         c_local = c_grid * (1 + epsilon * xi),   xi ~ N(0, 1)
   - Motivación: los agentes no tienen visión perfecta del terreno.
     Esta perturbación rompe la convergencia geométrica de los
     k-shortest paths entre pares distintos.
   - Parámetro: --noise_eps (default 0.15)

E) Congestión acumulativa (path avoidance)
   - Después de cada par, las celdas visitadas incrementan su costo
     en las aristas incidentes por un factor kappa.
   - Motivación: fenómeno de disuasión — las rutas muy transitadas
     se vuelven marginalmente menos atractivas para nuevas rutas.
   - Mantiene los corredores principales visibles (los primeros
     pares los usan), pero fuerza a los últimos pares a explorar
     alternativas.
   - Parámetro: --congestion_kappa (default 0.02)

Compromiso defendible: D es un modelo reconocido de "rough landscape"
en spatial cognition; E es congestion routing clásico en transport
networks. Ambos son estándar en modelado de movilidad.

Ejemplo de uso:
    python CA4.6.py --csv grilla_con_W_hidrica.csv --prefix v46
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label,
    maximum_filter,
    uniform_filter,
)
from tqdm import tqdm

plt.ioff()


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(description="Run V4.6 on the valley grid")
parser.add_argument("--csv", type=str, default="grilla_con_W_hidrica.csv", help="Input CSV path")
parser.add_argument("--prefix", type=str, default="v46", help="Prefix for output files")
parser.add_argument("--topK_mult", type=float, default=2.0, help="K = topK_mult * n_sites")
parser.add_argument("--nms_min_dist", type=int, default=5, help="Minimum distance between attractors")
parser.add_argument("--peak_quantile", type=float, default=0.92, help="Base quantile for local maxima")
parser.add_argument("--k_neighbors", type=int, default=6, help="Local neighbors per attractor")
parser.add_argument("--n_paths", type=int, default=4, help="Max number of simple paths per pair")
parser.add_argument("--snapshot_fracs", type=float, nargs="*", default=[0.25, 0.50, 0.75, 1.0])

# ── CAMBIO D: ruido perceptual ──────────────────────────────
parser.add_argument("--noise_eps", type=float, default=0.15,
                    help="Perceptual noise amplitude on edge costs (fix D). "
                         "0 = no noise, 0.1-0.2 = moderate, 0.3+ = very noisy.")
parser.add_argument("--noise_seed", type=int, default=42,
                    help="Random seed for noise reproducibility")

# ── CAMBIO E: congestión de caminos ─────────────────────────
parser.add_argument("--congestion_kappa", type=float, default=0.02,
                    help="Per-visit congestion increment for edge costs (fix E). "
                         "0 = no congestion, 0.02 = gentle, 0.05 = strong.")
parser.add_argument("--congestion_cap", type=float, default=2.0,
                    help="Maximum congestion multiplier (prevents runaway cost).")
args = parser.parse_args()

CSV_PATH = Path(args.csv)
PREFIX = args.prefix
if not CSV_PATH.exists():
    raise FileNotFoundError(f"No se encontró {CSV_PATH}")


def out(name: str, ext: str = "png") -> str:
    return f"{PREFIX}_{name}.{ext}"


# ============================================================
# 1. CARGA
# ============================================================

df = pd.read_csv(CSV_PATH)
required_cols = ["X", "Y", "altitud", "pendiente", "presencia_sitio_nuevo", "W_corridor"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas: {missing}")


def minmax(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


df["h_norm"] = minmax(df["altitud"])
df["p_norm"] = minmax(df["pendiente"])


# ============================================================
# 2. GRILLA
# ============================================================

def build_lattice(df, x_col="X", y_col="Y"):
    xs = np.sort(df[x_col].unique())
    ys = np.sort(df[y_col].unique())

    x_to_ix = {x: i for i, x in enumerate(xs)}
    y_to_iy = {y: i for i, y in enumerate(ys)}

    ny, nx = len(ys), len(xs)
    valid_mask = np.zeros((ny, nx), dtype=bool)

    df2 = df.reset_index(drop=True).copy()
    for _, row in df2.iterrows():
        iy = y_to_iy[row[y_col]]
        ix = x_to_ix[row[x_col]]
        valid_mask[iy, ix] = True

    return df2, {
        "xs": xs, "ys": ys,
        "x_to_ix": x_to_ix, "y_to_iy": y_to_iy,
        "shape": (ny, nx), "valid_mask": valid_mask,
    }


df, lattice = build_lattice(df)
valid_mask = lattice["valid_mask"]


def build_grid_from_column(df, lattice, col):
    grid = np.zeros(lattice["shape"], dtype=float)
    for _, row in df.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        grid[iy, ix] = row[col]
    return grid


h_grid = build_grid_from_column(df, lattice, "h_norm")
p_grid = build_grid_from_column(df, lattice, "p_norm")
w_grid = build_grid_from_column(df, lattice, "W_corridor")


def build_site_mask(df, lattice, arch_col="presencia_sitio_nuevo"):
    M = np.zeros(lattice["shape"], dtype=bool)
    arch = df[df[arch_col] == 1]
    for _, row in arch.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        M[iy, ix] = True
    return M


site_mask = build_site_mask(df, lattice)
N_SITES = int(site_mask.sum())
print(f"Sites found: {N_SITES}")


# ============================================================
# 3. CAMPOS DE PAISAJE
# ============================================================

def build_terrace_field(h_grid, p_grid, valid_mask, window=5):
    flat = 1.0 - p_grid
    low = 1.0 - h_grid
    flat_local = uniform_filter(flat, size=window, mode="nearest")
    T = 0.65 * flat + 0.35 * flat_local
    T = T * (0.8 + 0.2 * low)
    vals = T[valid_mask]
    Tn = np.zeros_like(T)
    Tn[valid_mask] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
    return Tn


T_grid = build_terrace_field(h_grid, p_grid, valid_mask, window=5)
df["T_terrace"] = [
    T_grid[lattice["y_to_iy"][row["Y"]], lattice["x_to_ix"][row["X"]]]
    for _, row in df.iterrows()
]


def compute_environment_and_cost(df, a_h=0.10, a_p=0.15, a_w=0.55, a_t=0.20,
                                 b_h=0.20, b_p=0.30, b_w=0.30, b_t=0.20):
    if not np.isclose(a_h + a_p + a_w + a_t, 1.0):
        raise ValueError("Los pesos de E deben sumar 1")
    if not np.isclose(b_h + b_p + b_w + b_t, 1.0):
        raise ValueError("Los pesos de c deben sumar 1")

    out_df = df.copy()
    out_df["E"] = (
        a_h * (1.0 - out_df["h_norm"]) +
        a_p * (1.0 - out_df["p_norm"]) +
        a_w * out_df["W_corridor"] +
        a_t * out_df["T_terrace"]
    )
    out_df["c"] = (
        b_h * out_df["h_norm"] +
        b_p * out_df["p_norm"] +
        b_w * (1.0 - out_df["W_corridor"]) +
        b_t * (1.0 - out_df["T_terrace"])
    )
    return out_df


# ============================================================
# 4. GRAFO — base (sin ruido ni congestión aún)
# ============================================================

def neighbors8(i, j, shape, valid_mask):
    ny, nx = shape
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx and valid_mask[ii, jj]:
                neigh.append((ii, jj))
    return neigh


def lattice_distance(i, j, ii, jj):
    return np.sqrt((ii - i) ** 2 + (jj - j) ** 2)


def build_graph(c_grid, valid_mask):
    """Grafo BASE. Los costos originales se guardan en attribute 'weight_base'.
    'weight' es el costo efectivo que se va actualizando por congestión."""
    G = nx.Graph()
    ny, nx_ = c_grid.shape
    for i in tqdm(range(ny), desc="Construyendo grafo"):
        for j in range(nx_):
            if not valid_mask[i, j]:
                continue
            G.add_node((i, j))
            for ii, jj in neighbors8(i, j, c_grid.shape, valid_mask):
                if G.has_edge((i, j), (ii, jj)):
                    continue
                edge_cost = 0.5 * (c_grid[i, j] + c_grid[ii, jj]) * lattice_distance(i, j, ii, jj)
                G.add_edge((i, j), (ii, jj),
                           weight=edge_cost,        # efectivo (se actualiza)
                           weight_base=edge_cost,   # original (referencia)
                           congestion_mult=1.0)      # factor actual de congestión
    return G


# ── FIX D: ruido perceptual por par ─────────────────────────
def apply_perceptual_noise(G, noise_eps, rng):
    """
    Sobrepone ruido multiplicativo gaussiano a los costos base.
    Cada llamada genera un paisaje 'percibido' ligeramente distinto.
    El ruido se aplica al weight_base multiplicado por el factor
    de congestión acumulado.
    """
    if noise_eps <= 0:
        # Sin ruido: weight = weight_base * congestion
        for u, v, data in G.edges(data=True):
            data["weight"] = data["weight_base"] * data["congestion_mult"]
        return

    for u, v, data in G.edges(data=True):
        xi = rng.standard_normal()
        perturbation = max(0.1, 1.0 + noise_eps * xi)  # evitar costos negativos
        data["weight"] = data["weight_base"] * data["congestion_mult"] * perturbation


# ── FIX E: congestión acumulativa ───────────────────────────
def apply_congestion_update(G, visited_edges, kappa, cap):
    """
    Incrementa el factor de congestión de las aristas visitadas.
    visited_edges es un dict {(u,v): count} con cuántas veces se pasó por ahí.
    """
    if kappa <= 0 or not visited_edges:
        return

    for edge, count in visited_edges.items():
        u, v = edge
        if G.has_edge(u, v):
            new_mult = G.edges[u, v]["congestion_mult"] + kappa * count
            G.edges[u, v]["congestion_mult"] = min(new_mult, cap)


def path_to_edges(path):
    """Convierte [(i,j), (i,j), ...] en [((u1,v1),(u2,v2)), ...]"""
    edges = {}
    for u, v in zip(path[:-1], path[1:]):
        key = tuple(sorted([u, v]))
        edges[key] = edges.get(key, 0) + 1
    return edges


# ============================================================
# 5. ATTRACTORS ADAPTATIVOS (igual que v4.5)
# ============================================================

def extract_local_maxima_adaptive(E_grid, valid_mask, peak_quantile=0.92, min_dist=5,
                                  n_bands=4, min_per_band=4):
    ny, nx = E_grid.shape
    E = np.where(valid_mask, E_grid, -np.inf)
    max_local = maximum_filter(E, size=2 * min_dist + 1, mode="constant", cval=-np.inf)
    peaks = (E == max_local) & valid_mask

    vals = E_grid[valid_mask]
    tau = np.quantile(vals, peak_quantile)
    peaks &= (E_grid >= tau)

    coords = np.argwhere(peaks)
    coords = sorted(coords, key=lambda ij: E_grid[ij[0], ij[1]], reverse=True)

    selected = []
    A_mask = np.zeros_like(E_grid, dtype=np.uint8)
    taken = np.zeros_like(E_grid, dtype=bool)

    def can_add(i, j):
        i0 = max(0, i - min_dist); i1 = min(ny, i + min_dist + 1)
        j0 = max(0, j - min_dist); j1 = min(nx, j + min_dist + 1)
        return not taken[i0:i1, j0:j1].any()

    for i, j in coords:
        if can_add(i, j):
            selected.append((int(i), int(j)))
            A_mask[i, j] = 1
            i0 = max(0, i - min_dist); i1 = min(ny, i + min_dist + 1)
            j0 = max(0, j - min_dist); j1 = min(nx, j + min_dist + 1)
            taken[i0:i1, j0:j1] = True

    band_edges = np.linspace(0, nx, n_bands + 1).astype(int)
    band_rows = []
    for b in range(n_bands):
        j0, j1 = band_edges[b], band_edges[b + 1]
        in_band = [(i, j) for i, j in selected if j0 <= j < j1]
        band_rows.append({
            "band": b + 1, "j0": int(j0), "j1": int(j1),
            "selected": len(in_band),
        })
        if len(in_band) >= min_per_band:
            continue

        band_valid = valid_mask[:, j0:j1]
        band_vals = E_grid[:, j0:j1][band_valid]
        if len(band_vals) == 0:
            continue
        tau_band = np.quantile(band_vals, min(0.98, peak_quantile + 0.03))
        cand = np.argwhere((E_grid[:, j0:j1] >= tau_band) & band_valid)
        cand = [(int(i), int(j + j0)) for i, j in cand]
        cand = sorted(cand, key=lambda ij: E_grid[ij[0], ij[1]], reverse=True)
        for i, j in cand:
            if len([1 for ii, jj in selected if j0 <= jj < j1]) >= min_per_band:
                break
            if can_add(i, j):
                selected.append((int(i), int(j)))
                A_mask[i, j] = 1
                i0 = max(0, i - min_dist); i1 = min(ny, i + min_dist + 1)
                jj0 = max(0, j - min_dist); jj1 = min(nx, j + min_dist + 1)
                taken[i0:i1, jj0:jj1] = True

    selected = sorted(selected, key=lambda ij: E_grid[ij[0], ij[1]], reverse=True)
    tau_global = min(E_grid[i, j] for i, j in selected) if selected else np.nan
    band_df = pd.DataFrame(band_rows)
    return A_mask, tau_global, selected, band_df


# ============================================================
# 6. PARES LOCALES
# ============================================================

def build_local_pairs(attractors, E_grid, k_neighbors=6, max_dist=None):
    pair_set = set()
    K = len(attractors)
    for a in range(K):
        i1, j1 = attractors[a]
        dists = []
        for b in range(K):
            if a == b:
                continue
            i2, j2 = attractors[b]
            d = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)
            if max_dist is None or d <= max_dist:
                dists.append((d, b))
        dists.sort(key=lambda x: x[0])
        for _, b in dists[:k_neighbors]:
            pair_set.add(tuple(sorted((a, b))))
    return sorted(pair_set)


# ============================================================
# 7. MÉTRICAS
# ============================================================

def threshold_field(F_norm, valid_mask, quantile=0.99):
    vals = F_norm[valid_mask]
    tau = np.quantile(vals, quantile)
    S = np.zeros_like(F_norm, dtype=np.uint8)
    S[(F_norm >= tau) & valid_mask] = 1
    return S, tau


def threshold_topK(F_norm, valid_mask, K):
    vals = F_norm[valid_mask]
    if K >= len(vals):
        tau = vals.min()
    elif K <= 0:
        tau = np.inf
    else:
        tau = np.partition(vals, -K)[-K]
    S = np.zeros_like(F_norm, dtype=np.uint8)
    S[(F_norm >= tau) & valid_mask] = 1
    return S, tau


def archaeological_overlap_exact(S_grid, lattice, df, arch_col="presencia_sitio_nuevo"):
    arch_cells = df[df[arch_col] == 1]
    if len(arch_cells) == 0:
        return np.nan
    hits = 0
    for _, row in arch_cells.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        hits += S_grid[iy, ix]
    return hits / len(arch_cells)


def buffered_overlap(S_grid, lattice, df, arch_col="presencia_sitio_nuevo", buffer_cells=1):
    arch_cells = df[df[arch_col] == 1]
    if len(arch_cells) == 0:
        return np.nan
    struct = np.ones((2 * buffer_cells + 1, 2 * buffer_cells + 1))
    S_dilated = binary_dilation(S_grid == 1, structure=struct)
    hits = 0
    for _, row in arch_cells.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        if S_dilated[iy, ix]:
            hits += 1
    return hits / len(arch_cells)


def nearest_distance_to_predicted(S_grid, lattice, df, arch_col="presencia_sitio_nuevo"):
    arch_cells = df[df[arch_col] == 1]
    pred_coords = np.argwhere(S_grid == 1)
    if len(arch_cells) == 0 or len(pred_coords) == 0:
        return np.nan, np.nan, []
    distances = []
    for _, row in arch_cells.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        dmin = np.min(np.sqrt((pred_coords[:, 0] - iy) ** 2 + (pred_coords[:, 1] - ix) ** 2))
        distances.append(float(dmin))
    return float(np.mean(distances)), float(np.median(distances)), distances


def archaeological_contrast_continuous(F_norm, lattice, df, arch_col="presencia_sitio_nuevo"):
    arch_vals = []
    non_arch_vals = []
    for _, row in df.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        if row[arch_col] == 1:
            arch_vals.append(F_norm[iy, ix])
        else:
            non_arch_vals.append(F_norm[iy, ix])
    if len(arch_vals) == 0 or len(non_arch_vals) == 0:
        return np.nan
    return np.mean(arch_vals) - np.mean(non_arch_vals)


def archaeological_contrast_zscore(F, site_mask, valid_mask):
    bg_mask = valid_mask & ~site_mask
    if site_mask.sum() == 0 or bg_mask.sum() == 0:
        return np.nan
    bg = F[bg_mask]
    mu_bg, sd_bg = bg.mean(), bg.std()
    if sd_bg < 1e-15:
        return np.nan
    return float((F[site_mask].mean() - mu_bg) / sd_bg)


def archaeological_ratio(F, site_mask, valid_mask):
    bg_mask = valid_mask & ~site_mask
    if site_mask.sum() == 0 or bg_mask.sum() == 0:
        return np.nan
    mu_bg = F[bg_mask].mean()
    if mu_bg < 1e-15:
        return np.nan
    return float(F[site_mask].mean() / mu_bg)


def auc_score(F, site_mask, valid_mask):
    bg_mask = valid_mask & ~site_mask
    if site_mask.sum() == 0 or bg_mask.sum() == 0:
        return np.nan
    pos = F[site_mask]
    neg = F[bg_mask]
    all_vals = np.concatenate([pos, neg])
    ranks = pd.Series(all_vals).rank(method="average").values
    n_pos = len(pos); n_neg = len(neg)
    sum_pos = ranks[:n_pos].sum()
    U = sum_pos - n_pos * (n_pos + 1) / 2
    auc = U / (n_pos * n_neg)
    return float(auc)


def cluster_sizes(S_grid, valid_mask):
    masked = np.where(valid_mask, S_grid, 0)
    structure = np.ones((3, 3), dtype=int)
    labeled, ncomp = label(masked, structure=structure)
    if ncomp == 0:
        return np.array([])
    return np.array([(labeled == k).sum() for k in range(1, ncomp + 1)])


# ============================================================
# 8. ENSAMBLE DE CAMINOS
# ============================================================

def k_simple_paths_weighted(G, source, target, n_paths=4):
    try:
        gen = nx.shortest_simple_paths(G, source=source, target=target, weight="weight")
        paths = []
        for _ in range(n_paths):
            try:
                paths.append(next(gen))
            except StopIteration:
                break
        return paths
    except nx.NetworkXNoPath:
        return []


def path_cost(G, path):
    return float(sum(G.edges[u, v]["weight"] for u, v in zip(path[:-1], path[1:])))


def path_length(path):
    L = 0.0
    for (u1, v1), (u2, v2) in zip(path[:-1], path[1:]):
        L += np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
    return float(L)


# ============================================================
# 9. CAMPO DE FIJACIÓN
# ============================================================

def build_fixation_field(F_norm, T_grid, valid_mask,
                         alpha_F=0.55, alpha_T=0.20, alpha_C=0.15, alpha_D=0.10,
                         sigma_c=1.2, window_d=5):
    F0 = np.where(valid_mask, F_norm, 0.0)
    C = gaussian_filter(F0, sigma=sigma_c)
    D = uniform_filter(F0, size=window_d, mode="nearest")

    def norm(arr):
        vals = arr[valid_mask]
        out = np.zeros_like(arr)
        out[valid_mask] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        return out

    Cn = norm(C); Dn = norm(D)
    L = alpha_F * F_norm + alpha_T * T_grid + alpha_C * Cn + alpha_D * Dn
    Ln = norm(L)
    return Ln, Cn, Dn


# ============================================================
# 10. RUN DEL MODELO — con D (ruido) y E (congestión)
# ============================================================

def run_model(config_name, df_base, lattice, valid_mask, site_mask,
              a_weights, b_weights, beta_path,
              peak_quantile=0.92, min_dist=5, k_neighbors=6,
              n_paths=4, topK_mult=2.0,
              noise_eps=0.15, noise_seed=42,
              congestion_kappa=0.02, congestion_cap=2.0):
    a_h, a_p, a_w, a_t = a_weights
    b_h, b_p, b_w, b_t = b_weights

    df_m = compute_environment_and_cost(
        df_base,
        a_h=a_h, a_p=a_p, a_w=a_w, a_t=a_t,
        b_h=b_h, b_p=b_p, b_w=b_w, b_t=b_t,
    )

    E_grid = build_grid_from_column(df_m, lattice, "E")
    c_grid = build_grid_from_column(df_m, lattice, "c")

    A_mask, tau_E, attractors, band_df = extract_local_maxima_adaptive(
        E_grid, valid_mask,
        peak_quantile=peak_quantile, min_dist=min_dist,
        n_bands=4, min_per_band=4,
    )

    print(f"[{config_name}] tau_E = {tau_E}")
    print(f"[{config_name}] n attractors = {len(attractors)}")

    G = build_graph(c_grid, valid_mask)
    print(f"[{config_name}] nodes = {G.number_of_nodes()} | n edges = {G.number_of_edges()}")
    print(f"[{config_name}] noise_eps = {noise_eps} | congestion_kappa = {congestion_kappa}")

    pair_candidates = build_local_pairs(attractors, E_grid, k_neighbors=k_neighbors)
    print(f"[{config_name}] candidate pairs = {len(pair_candidates)} (k_neighbors={k_neighbors})")

    # RNG para el ruido perceptual
    rng = np.random.default_rng(noise_seed)

    # Score pairs usando el grafo sin ruido ni congestión (costos base)
    # para determinar orden de procesamiento
    pair_records = []
    for a, b in tqdm(pair_candidates, desc=f"[{config_name}] scoring pairs"):
        i1, j1 = attractors[a]
        i2, j2 = attractors[b]
        try:
            # Usar weight_base limpio para scoring
            cmin = nx.shortest_path_length(
                G, source=(i1, j1), target=(i2, j2), weight="weight_base"
            )
            wab = np.exp(2.0 * E_grid[i1, j1]) * np.exp(2.0 * E_grid[i2, j2]) * np.exp(-3.0 * cmin)
            pair_records.append((a, b, wab, cmin))
        except nx.NetworkXNoPath:
            continue

    pair_records.sort(key=lambda r: r[2], reverse=True)
    print(f"[{config_name}] usable pairs = {len(pair_records)} | beta_path = {beta_path}")

    # ── Acumulación con D y E ─────────────────────────────
    traffic = np.zeros_like(E_grid, dtype=float)
    dynamics = []
    cumulative_cost = 0.0
    cumulative_eps = 0.0
    used_paths = 0

    snapshot_fracs = args.snapshot_fracs
    snapshot_steps = sorted(set(
        max(1, int(np.ceil(f * len(pair_candidates)))) for f in snapshot_fracs
    )) if pair_candidates else []
    snapshots_incremental = {}
    prev_snapshot = np.zeros_like(E_grid, dtype=float)

    K_eval = max(1, int(topK_mult * int(site_mask.sum())))

    # Diagnóstico: cuánto se dispersan las rutas
    diversity_records = []

    for idx, (a, b, wab, cmin) in enumerate(
        tqdm(pair_records, desc=f"[{config_name}] accumulating"), start=1
    ):
        i1, j1 = attractors[a]
        i2, j2 = attractors[b]

        # ── FIX D: perturbar el costo percibido para este par
        apply_perceptual_noise(G, noise_eps=noise_eps, rng=rng)

        # Buscar k-shortest paths en el paisaje perturbado + congestionado
        paths = k_simple_paths_weighted(G, (i1, j1), (i2, j2), n_paths=n_paths)
        if not paths:
            continue

        costs = np.array([path_cost(G, p) for p in paths], dtype=float)
        c0 = costs.min()
        boltz = np.exp(-beta_path * (costs - c0))
        boltz /= boltz.sum() + 1e-12

        pair_weight_total = np.exp(2.0 * E_grid[i1, j1]) * np.exp(2.0 * E_grid[i2, j2]) * np.exp(-3.0 * c0)

        # Diversidad de rutas (cuántas celdas distintas cubre el ensamble)
        cells_used = set()
        for path in paths:
            for cell in path:
                cells_used.add(cell)
        shortest_len = len(paths[0])
        diversity_records.append({
            "m": idx,
            "n_paths": len(paths),
            "mean_path_len": np.mean([len(p) for p in paths]),
            "shortest_len": shortest_len,
            "union_size": len(cells_used),
            "redundancy": shortest_len / len(cells_used) if len(cells_used) > 0 else 1.0,
        })

        # Agregar tráfico y acumular congestión por arista
        visited_edges = {}
        for path, cpath, pprob in zip(paths, costs, boltz):
            wpath = pair_weight_total * pprob
            for i, j in path:
                traffic[i, j] += wpath
            # aristas del path (para congestión)
            for u, v in zip(path[:-1], path[1:]):
                key = tuple(sorted([u, v]))
                visited_edges[key] = visited_edges.get(key, 0) + 1

            cumulative_cost += cpath
            cumulative_eps += cpath / (path_length(path) + 1e-12)
            used_paths += 1

        # ── FIX E: actualizar congestión de las aristas visitadas
        apply_congestion_update(G, visited_edges,
                                kappa=congestion_kappa,
                                cap=congestion_cap)

        # Métricas dinámicas
        Fsum = traffic.sum()
        F_norm = traffic / Fsum if Fsum > 0 else traffic.copy()
        L_field, C_field, D_field = build_fixation_field(F_norm, T_grid, valid_mask)

        Pm = 1.0 / (np.sum(F_norm ** 2) + 1e-12)
        vals = F_norm[valid_mask]
        vals = vals[vals > 0]
        Sm = -np.sum(vals * np.log(vals)) if len(vals) > 0 else 0.0

        S_eval_K, _ = threshold_topK(L_field, valid_mask, K=K_eval)
        Omega_topK = buffered_overlap(S_eval_K, lattice, df_base, buffer_cells=1)
        AUC_m = auc_score(L_field, site_mask, valid_mask)
        delta_z = archaeological_contrast_zscore(L_field, site_mask, valid_mask)
        ratio_sn = archaeological_ratio(L_field, site_mask, valid_mask)
        delta_raw = archaeological_contrast_continuous(L_field, lattice, df_base)

        dynamics.append({
            "m": idx,
            "pair_weight": wab,
            "participation": Pm,
            "entropy": Sm,
            "delta": delta_raw,
            "omega_topK": Omega_topK,
            "auc": AUC_m,
            "delta_zscore": delta_z,
            "ratio_sn": ratio_sn,
            "mean_cost": cumulative_cost / max(used_paths, 1),
            "mean_specific_cost": cumulative_eps / max(used_paths, 1),
        })

        if idx in snapshot_steps:
            current = traffic.copy()
            delta_snap = current - prev_snapshot
            delta_sum = delta_snap.sum()
            delta_norm = delta_snap / delta_sum if delta_sum > 0 else delta_snap.copy()
            snapshots_incremental[idx] = delta_norm
            prev_snapshot = current.copy()

    # finales
    Fsum = traffic.sum()
    F_norm = traffic / Fsum if Fsum > 0 else traffic.copy()
    L_field, C_field, D_field = build_fixation_field(F_norm, T_grid, valid_mask)

    S_topK, tau_topK = threshold_topK(L_field, valid_mask, K=K_eval)
    S_q99, tau_q99 = threshold_field(L_field, valid_mask, quantile=0.99)

    mean_d, median_d, _ = nearest_distance_to_predicted(S_topK, lattice, df_base)

    # Mapa final de congestión (para diagnóstico)
    congestion_grid = np.zeros_like(E_grid, dtype=float)
    for u, v, data in G.edges(data=True):
        mult = data.get("congestion_mult", 1.0)
        if mult > 1.0:
            congestion_grid[u[0], u[1]] = max(congestion_grid[u[0], u[1]], mult)
            congestion_grid[v[0], v[1]] = max(congestion_grid[v[0], v[1]], mult)

    return {
        "config": config_name,
        "df_model": df_m,
        "E_grid": E_grid, "c_grid": c_grid,
        "A_mask": A_mask, "attractors": attractors,
        "band_df": band_df,
        "G": G, "pair_records": pair_records,
        "traffic": traffic, "F_norm": F_norm,
        "L_field": L_field, "C_field": C_field, "D_field": D_field,
        "S_topK": S_topK, "S_q99": S_q99,
        "tau_topK": tau_topK, "tau_q99": tau_q99,
        "dynamics": pd.DataFrame(dynamics),
        "diversity": pd.DataFrame(diversity_records),
        "congestion_grid": congestion_grid,
        "auc": auc_score(L_field, site_mask, valid_mask),
        "delta_zscore": archaeological_contrast_zscore(L_field, site_mask, valid_mask),
        "ratio_sn": archaeological_ratio(L_field, site_mask, valid_mask),
        "omega_exact": archaeological_overlap_exact(S_topK, lattice, df_base),
        "omega_buf1": buffered_overlap(S_topK, lattice, df_base, buffer_cells=1),
        "omega_buf2": buffered_overlap(S_topK, lattice, df_base, buffer_cells=2),
        "mean_dist": mean_d, "median_dist": median_d,
        "n_loci": int(S_topK.sum()),
        "n_clusters": len(cluster_sizes(S_topK, valid_mask)),
        "largest_cluster": int(cluster_sizes(S_topK, valid_mask).max())
                           if len(cluster_sizes(S_topK, valid_mask)) > 0 else 0,
        "snapshots_incremental": snapshots_incremental,
    }


# ============================================================
# 11. CALIBRACIÓN — pesos que ganaron en v4.5 con distintos D+E
# ============================================================

# Usamos los pesos de cfg4_beta1.5 (ganador en v4.5) y barremos D y E
# Así aislamos el efecto de los nuevos parámetros.
BASE_A = (0.10, 0.20, 0.50, 0.20)
BASE_B = (0.18, 0.36, 0.26, 0.20)
BASE_BETA = 1.5

SEARCH_SPACE = [
    # Baseline (sin D ni E) — debe dar ~lo mismo que v4.5 cfg4
    {"name": "base_noDE",   "noise_eps": 0.00, "congestion_kappa": 0.00},
    # Solo D
    {"name": "D_only_low",  "noise_eps": 0.10, "congestion_kappa": 0.00},
    {"name": "D_only_mid",  "noise_eps": 0.15, "congestion_kappa": 0.00},
    {"name": "D_only_high", "noise_eps": 0.25, "congestion_kappa": 0.00},
    # Solo E
    {"name": "E_only_low",  "noise_eps": 0.00, "congestion_kappa": 0.02},
    {"name": "E_only_mid",  "noise_eps": 0.00, "congestion_kappa": 0.05},
    # D + E combinados
    {"name": "DE_moderate", "noise_eps": 0.15, "congestion_kappa": 0.03},
    {"name": "DE_strong",   "noise_eps": 0.25, "congestion_kappa": 0.05},
]

all_results = []
best = None
best_key = (-np.inf, -np.inf, np.inf)

for cfg in SEARCH_SPACE:
    print("\n" + "=" * 60)
    print(f"Running {cfg['name']}")
    print("=" * 60)
    res = run_model(
        config_name=cfg["name"],
        df_base=df, lattice=lattice,
        valid_mask=valid_mask, site_mask=site_mask,
        a_weights=BASE_A, b_weights=BASE_B,
        beta_path=BASE_BETA,
        peak_quantile=args.peak_quantile,
        min_dist=args.nms_min_dist,
        k_neighbors=args.k_neighbors,
        n_paths=args.n_paths,
        topK_mult=args.topK_mult,
        noise_eps=cfg["noise_eps"],
        noise_seed=args.noise_seed,
        congestion_kappa=cfg["congestion_kappa"],
        congestion_cap=args.congestion_cap,
    )

    diversity_mean = res["diversity"]["union_size"].mean() if len(res["diversity"]) > 0 else 0
    diversity_end  = res["diversity"]["union_size"].iloc[-1] if len(res["diversity"]) > 0 else 0

    all_results.append({
        "config": cfg["name"],
        "noise_eps": cfg["noise_eps"],
        "congestion_kappa": cfg["congestion_kappa"],
        "auc": res["auc"],
        "delta_zscore": res["delta_zscore"],
        "ratio_sn": res["ratio_sn"],
        "omega_exact": res["omega_exact"],
        "omega_buf1": res["omega_buf1"],
        "omega_buf2": res["omega_buf2"],
        "mean_dist": res["mean_dist"],
        "median_dist": res["median_dist"],
        "n_attractors": len(res["attractors"]),
        "n_pairs": len(res["pair_records"]),
        "n_loci": res["n_loci"],
        "n_clusters": res["n_clusters"],
        "largest_cluster": res["largest_cluster"],
        "diversity_mean_union": diversity_mean,
        "diversity_end_union": diversity_end,
    })

    key = (res["auc"], res["omega_buf1"],
           -res["median_dist"] if not np.isnan(res["median_dist"]) else -np.inf)
    if key > best_key:
        best_key = key
        best = res

summary_search = pd.DataFrame(all_results).sort_values(
    ["auc", "omega_buf1", "median_dist"], ascending=[False, False, True]
)
summary_search.to_csv(out("calibration_summary", "csv"), index=False, encoding="utf-8")
print("\n=== CALIBRATION SUMMARY ===")
print(summary_search.to_string(index=False))

if best is None:
    raise RuntimeError("No se pudo ajustar ningún modelo")


# ============================================================
# 12. CURVA omega vs K
# ============================================================

print("\n=== omega_buf vs K (análisis de sensibilidad) ===")
k_mults = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
k_curve = []
for km in k_mults:
    K = max(1, int(km * N_SITES))
    S_k, _ = threshold_topK(best["L_field"], valid_mask, K=K)
    k_curve.append({
        "K_mult":     km,
        "K":          K,
        "n_loci":     int(S_k.sum()),
        "omega_exact": archaeological_overlap_exact(S_k, lattice, df),
        "omega_buf1":  buffered_overlap(S_k, lattice, df, buffer_cells=1),
        "omega_buf2":  buffered_overlap(S_k, lattice, df, buffer_cells=2),
    })
k_curve_df = pd.DataFrame(k_curve)
k_curve_df.to_csv(out("omega_vs_K", "csv"), index=False)
print(k_curve_df.to_string(index=False))


# ============================================================
# 13. FIGURAS (básicamente iguales que v4.5)
# ============================================================

FIG_BG = "#ffffff"
INVALID_BG = "#f5f3ee"
ACTIVE_COLOR = "#1a1a1a"
SITE_COLOR = "#b22222"
SITE_EDGE = "#ffffff"
ATTR_COLOR = "#0a2540"
ATTR_HALO = "#ffffff"
GRAPH_COLOR = "#c8c6bf"
BG_LAND = "#eeeae2"

CMAP_CONT = plt.get_cmap("magma_r").copy()
CMAP_TRAFFIC = plt.get_cmap("inferno_r").copy()
CMAP_CONT.set_bad(INVALID_BG)
CMAP_TRAFFIC.set_bad(INVALID_BG)
CMAP_HILLSHADE = LinearSegmentedColormap.from_list(
    "hill", ["#ffffff", "#e8e5de", "#c4c0b5"], N=64
)

ATTR_SIZE, SITE_SIZE = 32, 50
TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 14, 10, 8


def _clean_axes(ax, title, title_pad=10):
    ax.set_title(title, fontsize=TITLE_SIZE, pad=title_pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_facecolor(FIG_BG)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6); spine.set_color("#888888")


def _synthetic_hillshade(E_grid, valid_mask, sigma=1.5):
    E = np.where(valid_mask, E_grid, np.nan)
    E_smooth = gaussian_filter(np.where(np.isnan(E), 0, E), sigma=sigma)
    dx, dy = np.gradient(E_smooth)
    azimuth = np.radians(315); altitude = np.radians(45)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = (np.cos(altitude) * np.cos(slope) +
          np.sin(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    hs = np.clip(hs, 0, 1)
    hs[~valid_mask] = np.nan
    return hs


def _plot_sites(ax, df, lattice, arch_col="presencia_sitio_nuevo", i0=0, j0=0):
    arch_cells = df[df[arch_col] == 1]
    if len(arch_cells) == 0: return 0
    xs, ys = [], []
    for _, row in arch_cells.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        xs.append(ix - j0); ys.append(iy - i0)
    ax.scatter(xs, ys, s=SITE_SIZE, marker="*", c=SITE_COLOR,
               edgecolors=SITE_EDGE, linewidths=0.8, zorder=7,
               label=f"Archaeological sites (n={len(xs)})")
    return len(xs)


def _plot_attractors(ax, A_mask, i0=0, j0=0, size=None):
    coords = np.argwhere(A_mask > 0)
    if len(coords) == 0: return 0
    ys, xs = coords[:, 0] - i0, coords[:, 1] - j0
    s = size if size is not None else ATTR_SIZE
    ax.scatter(xs, ys, s=s * 1.8, c=ATTR_HALO, edgecolors="none", zorder=5)
    ax.scatter(xs, ys, s=s, c=ATTR_COLOR, edgecolors=ATTR_HALO,
               linewidths=0.6, zorder=6, label=f"Attractors (n={len(coords)})")
    return len(coords)


def _plot_hillshade_bg(ax, hillshade, alpha=0.85):
    ax.imshow(np.ma.masked_invalid(hillshade), origin="lower",
              interpolation="bilinear", cmap=CMAP_HILLSHADE,
              alpha=alpha, zorder=1)


def plot_continuous_field(arr, valid_mask, title, savepath, df, lattice,
                          A_mask=None, show_sites=True, show_contours=True,
                          cmap=CMAP_CONT, cbar_label="Normalized value"):
    data = np.ma.masked_where(~valid_mask, arr)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200, facecolor=FIG_BG)
    ax.set_facecolor(BG_LAND)
    im = ax.imshow(data, origin="lower", interpolation="bilinear",
                   cmap=cmap, vmin=0, vmax=1, zorder=2)
    if show_contours:
        vals_valid = arr[valid_mask]
        if len(vals_valid) > 0:
            levels = np.quantile(vals_valid, [0.5, 0.75, 0.9])
            try:
                ax.contour(arr, levels=levels, colors=["#ffffff"],
                           linewidths=[0.3, 0.5, 0.7], alpha=0.4, zorder=3)
            except Exception:
                pass
    if A_mask is not None:
        _plot_attractors(ax, A_mask)
    if show_sites:
        _plot_sites(ax, df, lattice)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(0.4)
    _clean_axes(ax, title)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                  edgecolor="#aaaaaa", frameon=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_traffic(F_norm, valid_mask, title, savepath, df, lattice,
                 hillshade=None, A_mask=None, use_log=True, show_sites=True):
    F = np.where(valid_mask & (F_norm > 0), F_norm, np.nan)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200, facecolor=FIG_BG)
    ax.set_facecolor(BG_LAND)
    if hillshade is not None:
        _plot_hillshade_bg(ax, hillshade, alpha=0.85)
    if use_log and np.nanmax(F) > 0:
        vmin = np.nanmin(F[F > 0]); vmax = np.nanmax(F)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = "Normalized traffic (log)"
    else:
        norm = Normalize(vmin=0, vmax=np.nanmax(F) if np.nanmax(F) > 0 else 1)
        cbar_label = "Normalized traffic"
    im = ax.imshow(np.ma.masked_invalid(F), origin="lower",
                   interpolation="nearest", cmap=CMAP_TRAFFIC,
                   norm=norm, zorder=3)
    if A_mask is not None:
        _plot_attractors(ax, A_mask)
    if show_sites:
        _plot_sites(ax, df, lattice)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(0.4)
    _clean_axes(ax, title)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                  edgecolor="#aaaaaa", frameon=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_loci(S_grid, valid_mask, title, savepath, df, lattice,
              hillshade=None, A_mask=None, show_sites=True,
              active_label="Predicted locus"):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200, facecolor=FIG_BG)
    ax.set_facecolor(BG_LAND)
    if hillshade is not None:
        _plot_hillshade_bg(ax, hillshade, alpha=0.85)
    active = np.where(valid_mask & (S_grid > 0), 1.0, np.nan)
    ax.imshow(np.ma.masked_invalid(active), origin="lower",
              interpolation="nearest", cmap=ListedColormap([ACTIVE_COLOR]),
              vmin=0.5, vmax=1.5, zorder=3)
    if A_mask is not None:
        _plot_attractors(ax, A_mask)
    if show_sites:
        _plot_sites(ax, df, lattice)
    legend_handles = [Patch(facecolor=ACTIVE_COLOR, edgecolor="none", label=active_label)]
    if A_mask is not None and A_mask.sum() > 0:
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=ATTR_COLOR, markeredgecolor=ATTR_HALO,
                                     markersize=8, label="Attractor"))
    if show_sites:
        arch_n = int(df["presencia_sitio_nuevo"].sum())
        if arch_n > 0:
            legend_handles.append(Line2D([0], [0], marker="*", color="w",
                                         markerfacecolor=SITE_COLOR, markeredgecolor=SITE_EDGE,
                                         markersize=11, label=f"Archaeological site (n={arch_n})"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.92, edgecolor="#aaaaaa", frameon=True)
    _clean_axes(ax, title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_graph_support(valid_mask, A_mask, savepath, hillshade=None, node_step=2):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200, facecolor=FIG_BG)
    ax.set_facecolor(BG_LAND)
    if hillshade is not None:
        _plot_hillshade_bg(ax, hillshade, alpha=0.6)
    coords = np.argwhere(valid_mask)
    if node_step > 1:
        coords = coords[::node_step]
    ax.scatter(coords[:, 1], coords[:, 0], s=1.2, c=GRAPH_COLOR,
               alpha=0.55, linewidths=0, zorder=2)
    _plot_attractors(ax, A_mask)
    _clean_axes(ax, "Lattice support and adaptive attractors")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
              edgecolor="#aaaaaa", frameon=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_dynamics(dynamics_df, out_fn):
    series = [
        ("participation", "Participation ratio", r"$\mathcal{P}(m)$"),
        ("entropy", "Spatial entropy", r"$\mathcal{S}(m)$"),
        ("delta", "Archaeological contrast (raw)", r"$\Delta(m)$"),
        ("omega_topK", "Buffered overlap (top-K)", r"$\Omega_b^K(m)$"),
        ("auc", "AUC (ranking)", r"$\mathrm{AUC}(m)$"),
        ("delta_zscore", "Contrast z-score", r"$\Delta^*(m)$"),
        ("ratio_sn", "Site/background ratio", r"$R(m)$"),
        ("mean_cost", "Mean path cost", r"$\bar{\mathcal{C}}(m)$"),
        ("mean_specific_cost", "Mean specific cost", r"$\bar{\epsilon}(m)$"),
    ]
    for col, title, ylab in series:
        if col not in dynamics_df.columns or dynamics_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200, facecolor=FIG_BG)
        ax.plot(dynamics_df["m"], dynamics_df[col],
                linewidth=2.2, color="#2c3e50", zorder=3)
        ax.fill_between(dynamics_df["m"],
                        np.nanmin(dynamics_df[col].values),
                        dynamics_df[col],
                        color="#2c3e50", alpha=0.08, zorder=1)
        ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
        ax.set_xlabel("Accumulated pairs $m$", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylab, fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.25, color="#aaaaaa", linewidth=0.5, zorder=0)
        ax.set_facecolor(FIG_BG)
        ax.tick_params(labelsize=TICK_SIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6); spine.set_color("#888888")
        plt.tight_layout()
        plt.savefig(out_fn(col), dpi=300, bbox_inches="tight", facecolor=FIG_BG)
        plt.close(fig)


def plot_omega_vs_K(k_curve_df, savepath):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200, facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.plot(k_curve_df["K_mult"], k_curve_df["omega_exact"],
            "o-", color="#2c3e50", linewidth=2, markersize=7,
            label=r"$\Omega_{\mathrm{exact}}$", markerfacecolor="white")
    ax.plot(k_curve_df["K_mult"], k_curve_df["omega_buf1"],
            "s-", color="#b22222", linewidth=2, markersize=7,
            label=r"$\Omega_{\mathrm{buf},1}$", markerfacecolor="white")
    ax.plot(k_curve_df["K_mult"], k_curve_df["omega_buf2"],
            "^-", color="#0a2540", linewidth=2, markersize=7,
            label=r"$\Omega_{\mathrm{buf},2}$", markerfacecolor="white")
    ax.set_xlabel(r"$K / n_{\mathrm{sites}}$", fontsize=LABEL_SIZE + 1)
    ax.set_ylabel(r"Overlap with archaeological sites", fontsize=LABEL_SIZE + 1)
    ax.set_title("Sensitivity of overlap metric to threshold $K$",
                 fontsize=TITLE_SIZE, pad=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=10, edgecolor="#aaaaaa", loc="lower right")
    ax.set_ylim(0, 1.02)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6); spine.set_color("#888888")
    ax.tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_error_map_dual(L_field, valid_mask, site_mask, savepath,
                        lattice, df, n_sites,
                        K_mults=(2.0, 5.0),
                        hillshade=None):
    """
    Mapa de errores TP/FP/FN/TN en dos paneles para dos valores de K.
    - TP (verde): sitio real + predicho por el modelo
    - FP (azul): no sitio + predicho (candidato a prospección futura)
    - FN (rojo): sitio real + no predicho (donde el modelo falla)
    - TN (gris muy claro): no sitio + no predicho

    Los paneles comparan cómo cambia la predicción al relajar el umbral K.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), dpi=200, facecolor=FIG_BG)

    # Paleta: outside, TN, FP, FN, TP
    cmap_err = ListedColormap([
        INVALID_BG,     # 0: outside
        "#e8e6df",      # 1: TN gris claro
        "#2c6fbb",      # 2: FP azul
        "#b22222",      # 3: FN rojo
        "#2ca02c",      # 4: TP verde
    ])

    for ax, km in zip(axes, K_mults):
        K = max(1, int(km * n_sites))
        S_grid, _ = threshold_topK(L_field, valid_mask, K=K)

        pred = (S_grid == 1)
        TP = pred & site_mask
        FP = pred & ~site_mask & valid_mask
        FN = ~pred & site_mask
        TN = ~pred & ~site_mask & valid_mask

        cat_grid = np.zeros_like(S_grid, dtype=np.int8)
        cat_grid[TN] = 1
        cat_grid[FP] = 2
        cat_grid[FN] = 3
        cat_grid[TP] = 4

        n_tp, n_fp, n_fn = int(TP.sum()), int(FP.sum()), int(FN.sum())
        precision = n_tp / max(n_tp + n_fp, 1)
        recall    = n_tp / max(n_tp + n_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        ax.set_facecolor(BG_LAND)
        if hillshade is not None:
            _plot_hillshade_bg(ax, hillshade, alpha=0.7)

        ax.imshow(cat_grid, origin="lower", interpolation="nearest",
                  cmap=cmap_err, vmin=0, vmax=4, zorder=3, alpha=0.92)

        legend_handles = [
            Patch(facecolor="#2ca02c", edgecolor="white",
                  label=f"True Positive (n={n_tp})"),
            Patch(facecolor="#2c6fbb", edgecolor="white",
                  label=f"False Positive (n={n_fp})"),
            Patch(facecolor="#b22222", edgecolor="white",
                  label=f"False Negative (n={n_fn})"),
            Patch(facecolor="#e8e6df", edgecolor="#888888",
                  label="True Negative"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
                  framealpha=0.92, edgecolor="#aaaaaa", frameon=True)

        title = (f"K = {K} ({km:g}$\\times$ $n_{{\\mathrm{{sites}}}}$)   "
                 f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
        _clean_axes(ax, title, title_pad=8)

    fig.suptitle("Model prediction errors across thresholds",
                 fontsize=TITLE_SIZE + 1, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_calibration_matrix(summary_df, savepath):
    """Figura de matriz: AUC y omega_buf1 en función de (noise, kappa)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200, facecolor=FIG_BG)

    for ax, metric, title, ylab in [
        (axes[0], "auc", "AUC by configuration", "AUC"),
        (axes[1], "omega_buf1", "Overlap (K=2·n_sites, buf=1)", r"$\Omega_{\mathrm{buf},1}$")
    ]:
        ax.set_facecolor(FIG_BG)
        configs = summary_df["config"].tolist()
        values = summary_df[metric].values
        colors = ["#0a2540" if "base" in c else
                  "#2c3e50" if "D_only" in c else
                  "#b22222" if "E_only" in c else
                  "#d4762a" for c in configs]
        bars = ax.barh(configs, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xlabel(ylab, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6); spine.set_color("#888888")
        ax.tick_params(labelsize=TICK_SIZE)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


best_df = best["df_model"]
best_E = best["E_grid"]
best_F = best["F_norm"]
best_L = best["L_field"]
best_C = best["C_field"]
best_D = best["D_field"]
best_A = best["A_mask"]
best_S = best["S_topK"]
best_dyn = best["dynamics"]

hillshade = _synthetic_hillshade(best_E, valid_mask, sigma=1.5)

plot_continuous_field(best_E, valid_mask, "Environmental affinity $E$", out("E"), df, lattice,
                      A_mask=best_A, cbar_label="Affinity")
plot_continuous_field(T_grid, valid_mask, "Terrace suitability $T$", out("T"), df, lattice,
                      A_mask=best_A, cbar_label="Suitability")
plot_traffic(best_F, valid_mask, "Traffic field $F$", out("traffic"), df, lattice,
             hillshade=hillshade, A_mask=best_A, use_log=True)
plot_continuous_field(best_L, valid_mask, "Fixation field $L$", out("L"), df, lattice,
                      A_mask=best_A, cbar_label="Fixation suitability")
plot_continuous_field(best_C, valid_mask, "Convergence field $C$", out("C"), df, lattice,
                      A_mask=best_A, cbar_label="Convergence")
plot_continuous_field(best_D, valid_mask, "Persistence field $D$", out("D"), df, lattice,
                      A_mask=best_A, cbar_label="Persistence")
plot_loci(best_S, valid_mask,
          f"Thresholded loci (top-K, K={max(1, int(args.topK_mult * N_SITES))})",
          out("loci_topK"), df, lattice,
          hillshade=hillshade, A_mask=best_A, show_sites=True)
plot_graph_support(valid_mask, best_A, out("graph_support"),
                   hillshade=hillshade, node_step=2)
plot_dynamics(best_dyn, out)
plot_omega_vs_K(k_curve_df, out("omega_vs_K"))
plot_error_map_dual(
    best_L, valid_mask, site_mask,
    savepath=out("error_map"),
    lattice=lattice, df=df,
    n_sites=N_SITES,
    K_mults=(2.0, 5.0),
    hillshade=hillshade,
)
plot_calibration_matrix(summary_search, out("calibration_matrix"))


# ============================================================
# 14. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 60)
print("RESUMEN FINAL V4.6")
print("=" * 60)
print(f"Best config: {summary_search.iloc[0]['config']}")
print(summary_search.iloc[0].to_string())
print()
print(f"Cambios respecto a v4.5:")
print(f"  D) noise_eps = {summary_search.iloc[0]['noise_eps']}")
print(f"  E) congestion_kappa = {summary_search.iloc[0]['congestion_kappa']}")
print()
print(f"n sites:        {N_SITES}")
print(f"n attractors:   {len(best['attractors'])}")
print(f"n pairs:        {len(best['pair_records'])}")
print(f"n loci (topK):  {best['n_loci']}")
print(f"n clusters:     {best['n_clusters']}")
print(f"largest cluster:{best['largest_cluster']}")
print(f"AUC:            {best['auc']:.4f}")
print(f"delta_zscore:   {best['delta_zscore']:.3f}")
print(f"ratio_sn:       {best['ratio_sn']:.3f}")
print(f"omega_exact:    {best['omega_exact']:.3f}")
print(f"omega_buf1:     {best['omega_buf1']:.3f}")
print(f"omega_buf2:     {best['omega_buf2']:.3f}")
if not np.isnan(best['mean_dist']):
    print(f"mean site→locus:   {best['mean_dist']*250/1000:.2f} km")
    print(f"median site→locus: {best['median_dist']*250/1000:.2f} km")
print()
print(f"Diversity: mean union size = {best['diversity']['union_size'].mean():.1f} cells/pair")
print("\nFiguras guardadas y no abiertas.")