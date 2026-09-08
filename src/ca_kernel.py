"""
ca_kernel.py

Clean, importable kernel extracted from CA4_7.py.

Removes:
  - top-level execution
  - argparse
  - matplotlib / plotting
  - per-iteration metric computation (kept lean)

Provides:
  - build_state(csv_path) -> State with all static structure
  - run_one(state, eps, kappa, seed, ...) -> dict of observables
  - run_fete(state, n_random_pairs, seed, ...) -> FETE baseline

Deterministic reproducibility: same (eps, kappa, seed) tuple yields
identical results.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import networkx as nx
import pandas as pd
from scipy.ndimage import (
    uniform_filter, gaussian_filter, label, binary_dilation, maximum_filter
)


# =====================================================================
# LATTICE + FIELDS  (identical semantics to CA4_7.py)
# =====================================================================

def minmax(series):
    s = pd.Series(series).astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros_like(s.values))
    return (s - lo) / (hi - lo)


def build_lattice(df, x_col="X", y_col="Y"):
    xs = sorted(df[x_col].unique())
    ys = sorted(df[y_col].unique())
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


def build_grid_from_column(df, lattice, col):
    grid = np.zeros(lattice["shape"], dtype=float)
    for _, row in df.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        grid[iy, ix] = row[col]
    return grid


def build_site_mask(df, lattice, arch_col="presencia_sitio_nuevo"):
    M = np.zeros(lattice["shape"], dtype=bool)
    arch = df[df[arch_col] == 1]
    for _, row in arch.iterrows():
        iy = lattice["y_to_iy"][row["Y"]]
        ix = lattice["x_to_ix"][row["X"]]
        M[iy, ix] = True
    return M


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


def compute_environment_and_cost(df,
                                 a_h=0.10, a_p=0.20, a_w=0.50, a_t=0.20,
                                 b_h=0.18, b_p=0.36, b_w=0.26, b_t=0.20):
    df2 = df.copy()
    df2["E"] = (a_h * (1 - df2["h_norm"])
                + a_p * (1 - df2["p_norm"])
                + a_w * df2["W_corridor"]
                + a_t * df2["T_terrace"])
    df2["c"] = (b_h * df2["h_norm"]
                + b_p * df2["p_norm"]
                + b_w * (1 - df2["W_corridor"])
                + b_t * (1 - df2["T_terrace"]))
    return df2


# =====================================================================
# GRAPH + PERTURBATIONS
# =====================================================================

def neighbors8(i, j, shape, valid_mask):
    ny, nx = shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx and valid_mask[ii, jj]:
                yield (ii, jj)


def build_graph(c_grid, valid_mask):
    G = nx.Graph()
    ny, nx_ = c_grid.shape
    for i in range(ny):
        for j in range(nx_):
            if not valid_mask[i, j]:
                continue
            G.add_node((i, j))
            for ii, jj in neighbors8(i, j, c_grid.shape, valid_mask):
                if G.has_edge((i, j), (ii, jj)):
                    continue
                d = np.sqrt((ii - i) ** 2 + (jj - j) ** 2)
                w = 0.5 * (c_grid[i, j] + c_grid[ii, jj]) * d
                G.add_edge((i, j), (ii, jj),
                           weight=w,
                           weight_base=w,
                           congestion_mult=1.0)
    return G


def apply_perceptual_noise(G, noise_eps, rng):
    """Reset effective weight = base * congestion * (1 + eps * xi).

    If noise_eps == 0, effective weight = base * congestion only.
    Clipping keeps weights strictly positive.
    """
    if noise_eps <= 0:
        for u, v, data in G.edges(data=True):
            data["weight"] = data["weight_base"] * data["congestion_mult"]
        return
    for u, v, data in G.edges(data=True):
        xi = rng.standard_normal()
        perturbation = max(0.1, 1.0 + noise_eps * xi)
        data["weight"] = data["weight_base"] * data["congestion_mult"] * perturbation


def apply_congestion_update(G, visited_edges, kappa, cap):
    """Increment congestion multiplier on visited edges."""
    if kappa <= 0 or not visited_edges:
        return
    for edge, count in visited_edges.items():
        u, v = edge
        if G.has_edge(u, v):
            new_mult = G.edges[u, v]["congestion_mult"] + kappa * count
            G.edges[u, v]["congestion_mult"] = min(new_mult, cap)


# =====================================================================
# ATTRACTORS + PAIRS
# =====================================================================

def extract_local_maxima_adaptive(E_grid, valid_mask,
                                   peak_quantile=0.92, min_dist=5,
                                   n_bands=4, min_per_band=4):
    ny, nx = E_grid.shape
    E = np.where(valid_mask, E_grid, -np.inf)
    max_local = maximum_filter(E, size=2 * min_dist + 1,
                                mode="constant", cval=-np.inf)
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

    # Coverage: enforce min per longitudinal band
    band_edges = np.linspace(0, nx, n_bands + 1).astype(int)
    for b in range(n_bands):
        j0, j1 = band_edges[b], band_edges[b + 1]
        in_band = [(i, j) for i, j in selected if j0 <= j < j1]
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
            n_now = sum(1 for ii, jj in selected if j0 <= jj < j1)
            if n_now >= min_per_band:
                break
            if can_add(i, j):
                selected.append((int(i), int(j)))
                A_mask[i, j] = 1
                i0 = max(0, i - min_dist); i1 = min(ny, i + min_dist + 1)
                jj0 = max(0, j - min_dist); jj1 = min(nx, j + min_dist + 1)
                taken[i0:i1, jj0:jj1] = True

    selected = sorted(selected, key=lambda ij: E_grid[ij[0], ij[1]], reverse=True)
    tau_global = min(E_grid[i, j] for i, j in selected) if selected else np.nan
    return A_mask, tau_global, selected


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


# =====================================================================
# PATHS
# =====================================================================

def k_simple_paths_weighted(G, source, target, n_paths=4):
    try:
        gen = nx.shortest_simple_paths(G, source=source, target=target,
                                        weight="weight")
        out = []
        for _ in range(n_paths):
            try:
                out.append(next(gen))
            except StopIteration:
                break
        return out
    except nx.NetworkXNoPath:
        return []


def path_cost(G, path):
    return float(sum(G.edges[u, v]["weight"]
                     for u, v in zip(path[:-1], path[1:])))


# =====================================================================
# METRICS + FIXATION FIELD
# =====================================================================

def build_fixation_field(F_norm, T_grid, valid_mask,
                          alpha_F=0.55, alpha_T=0.20,
                          alpha_C=0.15, alpha_D=0.10,
                          sigma=1.0, window=5):
    """L = alpha_F F + alpha_T T + alpha_C C + alpha_D D, all in [0,1]."""
    F0 = np.where(valid_mask, F_norm, 0.0)
    C = gaussian_filter(F0, sigma=sigma, mode="nearest")
    D = uniform_filter(F0, size=window, mode="nearest")

    def norm(arr):
        vals = arr[valid_mask]
        out = np.zeros_like(arr)
        out[valid_mask] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        return out

    Cn = norm(C); Dn = norm(D)
    L = alpha_F * F_norm + alpha_T * T_grid + alpha_C * Cn + alpha_D * Dn
    return norm(L), Cn, Dn


def threshold_topK(F_norm, valid_mask, K):
    valid_vals = F_norm[valid_mask]
    if K >= valid_vals.size:
        tau = valid_vals.min()
    elif K <= 0:
        tau = np.inf
    else:
        tau = np.partition(valid_vals, -K)[-K]
    S = (F_norm >= tau) & valid_mask
    return S, tau


def cluster_sizes(S_grid, valid_mask):
    structure = np.ones((3, 3), dtype=int)
    labeled, _ = label(S_grid, structure=structure)
    if labeled.max() == 0:
        return np.array([]), None
    sizes = np.bincount(labeled.ravel())[1:]
    return sizes, labeled


def auc_score(F, site_mask, valid_mask):
    pos = F[site_mask]
    neg = F[valid_mask & ~site_mask]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    all_vals = np.concatenate([pos, neg])
    ranks = pd.Series(all_vals).rank().values
    pos_ranks = ranks[:len(pos)]
    U = pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2
    return float(U / (len(pos) * len(neg)))


# =====================================================================
# STATE (built once per CSV, reused across runs)
# =====================================================================

@dataclass
class State:
    df: pd.DataFrame
    lattice: dict
    valid_mask: np.ndarray
    site_mask: np.ndarray
    h_grid: np.ndarray
    p_grid: np.ndarray
    w_grid: np.ndarray
    T_grid: np.ndarray
    E_grid: np.ndarray
    c_grid: np.ndarray
    A_mask: np.ndarray
    attractors: list
    pair_candidates: list
    n_sites: int
    G_template: nx.Graph


def build_state(csv_path,
                a_weights=(0.10, 0.20, 0.50, 0.20),
                b_weights=(0.18, 0.36, 0.26, 0.20),
                peak_quantile=0.92, nms_min_dist=5, k_neighbors=6):
    df = pd.read_csv(csv_path)
    if "h_norm" not in df.columns:
        df["h_norm"] = minmax(df["altitud"])
    if "p_norm" not in df.columns:
        df["p_norm"] = minmax(df["pendiente"])

    df, lattice = build_lattice(df)
    valid_mask = lattice["valid_mask"]

    h_grid = build_grid_from_column(df, lattice, "h_norm")
    p_grid = build_grid_from_column(df, lattice, "p_norm")
    w_grid = build_grid_from_column(df, lattice, "W_corridor")
    T_grid = build_terrace_field(h_grid, p_grid, valid_mask, window=5)

    df["T_terrace"] = [
        T_grid[lattice["y_to_iy"][row["Y"]], lattice["x_to_ix"][row["X"]]]
        for _, row in df.iterrows()
    ]

    site_mask = build_site_mask(df, lattice)
    n_sites = int(site_mask.sum())

    df_m = compute_environment_and_cost(df, *a_weights, *b_weights)
    E_grid = build_grid_from_column(df_m, lattice, "E")
    c_grid = build_grid_from_column(df_m, lattice, "c")

    A_mask, _, attractors = extract_local_maxima_adaptive(
        E_grid, valid_mask,
        peak_quantile=peak_quantile, min_dist=nms_min_dist,
        n_bands=4, min_per_band=4)

    G_template = build_graph(c_grid, valid_mask)
    pair_candidates = build_local_pairs(attractors, E_grid,
                                         k_neighbors=k_neighbors)

    return State(
        df=df, lattice=lattice, valid_mask=valid_mask, site_mask=site_mask,
        h_grid=h_grid, p_grid=p_grid, w_grid=w_grid, T_grid=T_grid,
        E_grid=E_grid, c_grid=c_grid,
        A_mask=A_mask, attractors=attractors,
        pair_candidates=pair_candidates,
        n_sites=n_sites, G_template=G_template,
    )


# =====================================================================
# ONE RUN OF THE FULL CA MODEL
# =====================================================================

def run_one(state: State,
            eps: float, kappa: float, seed: int,
            beta_path: float = 1.5,
            n_paths: int = 4,
            topK_mult: float = 2.0,
            congestion_cap: float = 2.0):
    """Run the CA model once. Returns dict of observables.

    m1: fraction of top-K cells in largest connected component
    m2: fraction of total normalised traffic within largest component
    m3: exp(-S) where S is Shannon entropy of normalized F over Omega
    """
    G = deepcopy(state.G_template)
    rng = np.random.default_rng(seed)

    # Score pairs by importance (deterministic, uses base weights)
    pair_records = []
    for a, b in state.pair_candidates:
        i1, j1 = state.attractors[a]
        i2, j2 = state.attractors[b]
        try:
            cmin = nx.shortest_path_length(
                G, source=(i1, j1), target=(i2, j2), weight="weight_base")
            wab = (np.exp(2.0 * state.E_grid[i1, j1])
                   * np.exp(2.0 * state.E_grid[i2, j2])
                   * np.exp(-3.0 * cmin))
            pair_records.append((a, b, wab, cmin))
        except nx.NetworkXNoPath:
            continue

    pair_records.sort(key=lambda r: r[2], reverse=True)
    traffic = np.zeros_like(state.E_grid, dtype=float)

    for a, b, wab, _ in pair_records:
        i1, j1 = state.attractors[a]
        i2, j2 = state.attractors[b]
        apply_perceptual_noise(G, eps, rng)
        paths = k_simple_paths_weighted(G, (i1, j1), (i2, j2),
                                         n_paths=n_paths)
        if not paths:
            continue
        costs = np.array([path_cost(G, p) for p in paths], dtype=float)
        c0 = costs.min()
        boltz = np.exp(-beta_path * (costs - c0))
        boltz /= boltz.sum() + 1e-12

        pair_weight = (np.exp(2.0 * state.E_grid[i1, j1])
                       * np.exp(2.0 * state.E_grid[i2, j2])
                       * np.exp(-3.0 * c0))

        visited_edges = {}
        for path, pprob in zip(paths, boltz):
            wpath = pair_weight * pprob
            for i, j in path:
                traffic[i, j] += wpath
            for u, v in zip(path[:-1], path[1:]):
                key = tuple(sorted([u, v]))
                visited_edges[key] = visited_edges.get(key, 0) + 1

        apply_congestion_update(G, visited_edges,
                                kappa=kappa, cap=congestion_cap)

    Fsum = traffic.sum()
    F_norm = traffic / Fsum if Fsum > 0 else traffic.copy()

    L_field, _, _ = build_fixation_field(F_norm, state.T_grid,
                                          state.valid_mask)

    K_eval = max(1, int(topK_mult * state.n_sites))
    S_topK, _ = threshold_topK(L_field, state.valid_mask, K=K_eval)

    sizes, labeled = cluster_sizes(S_topK, state.valid_mask)
    lcc = int(sizes.max()) if len(sizes) else 0
    n_thr = int(S_topK.sum())
    m1 = lcc / n_thr if n_thr > 0 else 0.0

    if lcc > 0 and labeled is not None:
        lcc_label = int(sizes.argmax()) + 1
        mask_lcc = (labeled == lcc_label)
        m2 = float(F_norm[mask_lcc].sum())
    else:
        m2 = 0.0

    vals = F_norm[state.valid_mask]
    vals = vals[vals > 0]
    if len(vals) > 0:
        ps = vals / vals.sum()
        # Shannon entropy S = -sum(p log p) is non-negative for a
        # probability distribution.  Concentration index m3 = exp(-S)
        # is in [1/N, 1]:  1/N = uniform,  1 = single-cell delta.
        # Earlier drafts computed exp(-(-sum p log p)) = exp(S), which
        # inflated the value beyond 1.  Fixed here.
        S = -np.sum(ps * np.log(ps))     # entropy, positive
        m3 = float(np.exp(-S))            # concentration in [1/N, 1]
    else:
        m3 = 0.0

    auc = auc_score(L_field, state.site_mask, state.valid_mask)

    return {
        "eps": float(eps), "kappa": float(kappa), "seed": int(seed),
        "lcc_size": lcc,
        "n_threshold": n_thr,
        "m1": float(m1), "m2": float(m2), "m3": float(m3),
        "auc": float(auc),
        "n_clusters": int(len(sizes)),
        "F_norm_sum_check": float(F_norm.sum()),
    }


# =====================================================================
# FETE BASELINE (for reviewer 2)
#
# From-Everywhere-to-Everywhere: single deterministic LCP between many
# randomly sampled endpoint pairs on the SAME cost surface, aggregated
# into a traffic field and then converted to L via the same fixation
# formula. This is the standard baseline in movement archaeology.
# =====================================================================

def run_fete(state: State, n_random_pairs: int, seed: int,
             topK_mult: float = 2.0):
    """Baseline: FETE with random endpoint pairs, deterministic LCP,
    no noise or congestion. Returns the same observables as run_one so
    they can be compared point-to-point.
    """
    G = state.G_template  # no copy: we don't mutate
    rng = np.random.default_rng(seed)

    valid_cells = np.argwhere(state.valid_mask)
    if len(valid_cells) < 2:
        raise ValueError("Not enough valid cells")

    traffic = np.zeros_like(state.E_grid, dtype=float)
    n_success = 0
    for _ in range(n_random_pairs):
        # sample two distinct cells
        idx1, idx2 = rng.integers(0, len(valid_cells), size=2)
        while idx1 == idx2:
            idx2 = rng.integers(0, len(valid_cells))
        p1 = tuple(valid_cells[idx1])
        p2 = tuple(valid_cells[idx2])
        try:
            path = nx.shortest_path(G, source=p1, target=p2,
                                     weight="weight_base")
            for i, j in path:
                traffic[i, j] += 1.0
            n_success += 1
        except nx.NetworkXNoPath:
            continue

    Fsum = traffic.sum()
    F_norm = traffic / Fsum if Fsum > 0 else traffic.copy()
    L_field, _, _ = build_fixation_field(F_norm, state.T_grid,
                                          state.valid_mask)

    K_eval = max(1, int(topK_mult * state.n_sites))
    S_topK, _ = threshold_topK(L_field, state.valid_mask, K=K_eval)

    sizes, labeled = cluster_sizes(S_topK, state.valid_mask)
    lcc = int(sizes.max()) if len(sizes) else 0
    n_thr = int(S_topK.sum())
    m1 = lcc / n_thr if n_thr > 0 else 0.0
    if lcc > 0 and labeled is not None:
        lcc_label = int(sizes.argmax()) + 1
        m2 = float(F_norm[labeled == lcc_label].sum())
    else:
        m2 = 0.0
    vals = F_norm[state.valid_mask]
    vals = vals[vals > 0]
    if len(vals) > 0:
        ps = vals / vals.sum()
        # Shannon entropy S = -sum(p log p) is non-negative for a
        # probability distribution.  Concentration index m3 = exp(-S)
        # is in [1/N, 1]:  1/N = uniform,  1 = single-cell delta.
        # Earlier drafts computed exp(-(-sum p log p)) = exp(S), which
        # inflated the value beyond 1.  Fixed here.
        S = -np.sum(ps * np.log(ps))     # entropy, positive
        m3 = float(np.exp(-S))            # concentration in [1/N, 1]
    else:
        m3 = 0.0
    auc = auc_score(L_field, state.site_mask, state.valid_mask)

    return {
        "method": "fete",
        "n_random_pairs": n_random_pairs,
        "n_successful_paths": n_success,
        "seed": int(seed),
        "lcc_size": lcc,
        "n_threshold": n_thr,
        "m1": float(m1), "m2": float(m2), "m3": float(m3),
        "auc": float(auc),
        "n_clusters": int(len(sizes)),
    }
