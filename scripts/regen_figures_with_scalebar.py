#!/usr/bin/env python3
"""
regen_figures_with_scalebar.py

Regenerates Figures 3 and 5 with scalebar + north arrow, addressing
JCAA reviewer 1's request. Also produces Figure 4 (two-panel) with
the same scalebar, replacing regen_figure4.py's previous output.

Reads state from state_cache.pkl and (for Fig 5) the phase1 canonical
seed at eps=0.16, kappa=0.03 to derive L.

Outputs (all with scalebar of 5 km + north arrow):
    v47_E.png                  Fig 3: environmental affinity
    figure4_twopanel.png       Fig 4: deterministic vs canonical
    vz1_zones_L_mean.png       Fig 5: zonal Lbar over K-means

Usage:
    python regen_figures_with_scalebar.py
"""
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ca_kernel import (
    apply_perceptual_noise, apply_congestion_update,
    k_simple_paths_weighted, path_cost, build_fixation_field,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_bbox(valid_mask):
    """Return (r0, r1, c0, c1) tight bounding box of valid cells."""
    rows = np.where(valid_mask.any(axis=1))[0]
    cols = np.where(valid_mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def add_scalebar_and_north(ax, ny_c, nx_c, cell_km=0.25,
                            scale_km=5, color='white',
                            place_scale='bottom-left',
                            place_north='top-right'):
    """Add a scalebar and north arrow with semi-transparent backdrop
    for visibility on any underlying color."""
    cells = int(round(scale_km / cell_km))

    # ---- Scalebar ----
    if place_scale == 'bottom-left':
        xf, yf = 0.04, 0.06
    elif place_scale == 'bottom-right':
        xf, yf = 1 - 0.04 - cells / nx_c, 0.06
    elif place_scale == 'top-left':
        xf, yf = 0.04, 0.94
    else:
        xf, yf = 1 - 0.04 - cells / nx_c, 0.94

    x0 = int(xf * nx_c); x1 = x0 + cells
    y0 = int(yf * ny_c)

    # Backdrop (semi-transparent white/black box)
    bg_color = 'black' if color == 'white' else 'white'
    pad_x = int(0.015 * nx_c); pad_y = int(0.025 * ny_c)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (x0 - pad_x, y0 - pad_y),
        (x1 - x0) + 2 * pad_x, 3 * pad_y,
        facecolor=bg_color, alpha=0.55,
        edgecolor='none', zorder=9))

    ax.plot([x0, x1], [y0, y0], color=color, lw=4.0,
            solid_capstyle='butt', zorder=10)
    # Tick ends
    tick_h = int(0.012 * ny_c)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h],
            color=color, lw=2.5, zorder=10)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h],
            color=color, lw=2.5, zorder=10)
    ax.text((x0 + x1) / 2, y0 + int(0.030 * ny_c),
             f'{scale_km} km', color=color, ha='center', va='bottom',
             fontsize=10, fontweight='bold', zorder=11)

    # ---- North arrow ----
    if place_north == 'top-right':
        xn = int(0.94 * nx_c); yn_tail = int(0.94 * ny_c)
    elif place_north == 'top-left':
        xn = int(0.06 * nx_c); yn_tail = int(0.94 * ny_c)
    elif place_north == 'bottom-right':
        xn = int(0.94 * nx_c); yn_tail = int(0.20 * ny_c)
    else:
        xn = int(0.06 * nx_c); yn_tail = int(0.20 * ny_c)

    yn_head = yn_tail - int(0.10 * ny_c)

    # Backdrop for N arrow
    box_w = int(0.05 * nx_c)
    box_h = int(0.14 * ny_c)
    ax.add_patch(Rectangle(
        (xn - box_w // 2, yn_head - int(0.02 * ny_c)),
        box_w, box_h,
        facecolor=bg_color, alpha=0.55,
        edgecolor='none', zorder=9))

    ax.annotate('', xy=(xn, yn_tail), xytext=(xn, yn_head),
                 arrowprops=dict(arrowstyle='-|>', color=color, lw=2.4),
                 zorder=10)
    ax.text(xn, yn_head - int(0.015 * ny_c), 'N', color=color,
             ha='center', va='top', fontsize=12, fontweight='bold',
             zorder=11)


def compute_traffic_field(state, eps, kappa, seed):
    """Run one CA pass; return raw (un-normalised) traffic field."""
    G = deepcopy(state.G_template)
    rng = np.random.default_rng(seed)
    pair_records = []
    for a, b in state.pair_candidates:
        i1, j1 = state.attractors[a]
        i2, j2 = state.attractors[b]
        try:
            cmin = nx.shortest_path_length(
                G, source=(i1, j1), target=(i2, j2),
                weight="weight_base")
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
        paths = k_simple_paths_weighted(G, (i1, j1), (i2, j2), n_paths=4)
        if not paths:
            continue
        costs = np.array([path_cost(G, p) for p in paths], dtype=float)
        c0 = costs.min()
        boltz = np.exp(-1.5 * (costs - c0))
        boltz /= boltz.sum() + 1e-12
        pw = (np.exp(2.0 * state.E_grid[i1, j1])
              * np.exp(2.0 * state.E_grid[i2, j2])
              * np.exp(-3.0 * c0))
        visited = {}
        for path, pprob in zip(paths, boltz):
            wpath = pw * pprob
            for i, j in path:
                traffic[i, j] += wpath
            for u, v in zip(path[:-1], path[1:]):
                key = tuple(sorted([u, v]))
                visited[key] = visited.get(key, 0) + 1
        apply_congestion_update(G, visited, kappa=kappa, cap=2.0)
    return traffic


# ---------------------------------------------------------------------
# FIGURE 3: environmental affinity + attractors + sites
# ---------------------------------------------------------------------

def make_fig3(state, out_path):
    E = state.E_grid.copy()
    Em = np.where(state.valid_mask, E, np.nan)
    r0, r1, c0, c1 = get_bbox(state.valid_mask)
    crop = Em[r0:r1, c0:c1]
    ny_c, nx_c = crop.shape

    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(crop, cmap='magma', origin='lower',
                    interpolation='nearest', vmin=0, vmax=1)

    # Attractors
    for i, j in state.attractors:
        if r0 <= i < r1 and c0 <= j < c1:
            ax.plot(j - c0, i - r0, 'o', markersize=8,
                    markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=1.0,
                    zorder=5)

    # Sites
    site_ii, site_jj = np.where(state.site_mask)
    for i, j in zip(site_ii, site_jj):
        if r0 <= i < r1 and c0 <= j < c1:
            ax.plot(j - c0, i - r0, '*', markersize=8,
                    markerfacecolor='#ffe45c',
                    markeredgecolor='black', markeredgewidth=0.5,
                    zorder=6)

    add_scalebar_and_north(ax, ny_c, nx_c,
                            place_scale='bottom-left',
                            place_north='top-right')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_color('0.4')
    ax.set_title(r'Environmental affinity $E$', fontsize=12)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label=f'Attractors (n={len(state.attractors)})',
                markerfacecolor='white', markeredgecolor='black',
                markersize=8, markeredgewidth=1.0),
        Line2D([0], [0], marker='*', color='w', label=f'Sites (n={state.n_sites})',
                markerfacecolor='#ffe45c', markeredgecolor='black',
                markersize=10, markeredgewidth=0.5),
    ]
    ax.legend(handles=legend_elems, loc='lower right',
              framealpha=0.85, fontsize=9)

    plt.subplots_adjust(left=0.02, right=0.9, top=0.94, bottom=0.03)
    cbar_ax = fig.add_axes([0.905, 0.15, 0.014, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Affinity value', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------
# FIGURE 4: deterministic vs canonical traffic field, side-by-side
# ---------------------------------------------------------------------

def make_fig4(state, out_path, seed=42):
    print("  computing deterministic baseline (eps=0, kappa=0)...")
    t0 = time.time()
    F_det = compute_traffic_field(state, eps=0.0, kappa=0.0, seed=seed)
    print(f"    done in {time.time()-t0:.1f}s")

    print("  computing canonical (eps=0.15, kappa=0.03)...")
    t0 = time.time()
    F_can = compute_traffic_field(state, eps=0.15, kappa=0.03, seed=seed)
    print(f"    done in {time.time()-t0:.1f}s")

    F_det_n = F_det / (F_det.sum() + 1e-12)
    F_can_n = F_can / (F_can.sum() + 1e-12)

    def masked(arr):
        return np.where(state.valid_mask & (arr > 0), arr, np.nan)

    F_det_p = masked(F_det_n); F_can_p = masked(F_can_n)
    all_vals = np.concatenate([F_det_p[np.isfinite(F_det_p)],
                                F_can_p[np.isfinite(F_can_p)]])
    vmin = all_vals.min(); vmax = all_vals.max()

    r0, r1, c0, c1 = get_bbox(state.valid_mask)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, F, title, add_ns in zip(
        axes,
        [F_det_p, F_can_p],
        [r'(a) Deterministic  ($\varepsilon = 0$, $\kappa = 0$)',
         r'(b) Canonical  ($\varepsilon = 0.15$, $\kappa = 0.03$)'],
        [True, False],
    ):
        crop = F[r0:r1, c0:c1]
        ny_c, nx_c = crop.shape
        im = ax.imshow(crop, cmap='magma_r', origin='lower',
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       interpolation='nearest')
        for (i, j) in state.attractors:
            if r0 <= i < r1 and c0 <= j < c1:
                ax.plot(j - c0, i - r0, 'o', markersize=5,
                        markerfacecolor='white', markeredgecolor='black',
                        markeredgewidth=0.6, zorder=5)
        site_ii, site_jj = np.where(state.site_mask)
        for i, j in zip(site_ii, site_jj):
            if r0 <= i < r1 and c0 <= j < c1:
                ax.plot(j - c0, i - r0, '*', markersize=6,
                        markerfacecolor='crimson', markeredgecolor='white',
                        markeredgewidth=0.4, zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color('0.4')
        if add_ns:
            add_scalebar_and_north(ax, ny_c, nx_c,
                                    place_scale='bottom-left',
                                    place_north='top-right')

    fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,
                        wspace=0.05)
    cbar_ax = fig.add_axes([0.915, 0.15, 0.012, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Normalized traffic (log)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------
# FIGURE 5: zonal Lbar over K-means partition
# ---------------------------------------------------------------------

def make_fig5(state, out_path, seed=42, n_zones=20):
    print("  computing L field for zonal aggregation...")
    F = compute_traffic_field(state, eps=0.15, kappa=0.03, seed=seed)
    F_norm = F / (F.sum() + 1e-12)
    L_field, _, _ = build_fixation_field(F_norm, state.T_grid,
                                          state.valid_mask)

    valid = state.valid_mask
    yy, xx = np.where(valid)
    h_at = state.h_grid[valid]
    p_at = state.p_grid[valid]
    X = np.column_stack([xx, yy, h_at, p_at]).astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    # Assign labels back to grid
    zone_grid = np.full(state.E_grid.shape, -1, dtype=int)
    zone_grid[valid] = labels
    # Zonal mean L
    L_at_valid = L_field[valid]
    L_mean = np.array([L_at_valid[labels == z].mean() for z in range(n_zones)])
    # Site count per zone
    site_at_valid = state.site_mask[valid].astype(int)
    site_count = np.array([site_at_valid[labels == z].sum()
                            for z in range(n_zones)])

    # Build zonal Lbar image
    lbar_img = np.full(state.E_grid.shape, np.nan)
    for z in range(n_zones):
        mask = (zone_grid == z)
        lbar_img[mask] = L_mean[z]

    r0, r1, c0, c1 = get_bbox(valid)
    crop = lbar_img[r0:r1, c0:c1]
    ny_c, nx_c = crop.shape

    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(crop, cmap='YlOrRd', origin='lower',
                    interpolation='nearest')

    # Compute centroids and annotate site counts
    for z in range(n_zones):
        mask = (zone_grid == z)
        if not mask.any():
            continue
        iy, ix = np.where(mask)
        cy = iy.mean() - r0
        cx = ix.mean() - c0
        if site_count[z] > 0:
            ax.text(cx, cy, str(int(site_count[z])),
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.2',
                              facecolor='white', edgecolor='black',
                              linewidth=0.8))

    # Sites as small stars
    site_ii, site_jj = np.where(state.site_mask)
    for i, j in zip(site_ii, site_jj):
        if r0 <= i < r1 and c0 <= j < c1:
            ax.plot(j - c0, i - r0, '*', markersize=5,
                    markerfacecolor='#c73030', markeredgecolor='white',
                    markeredgewidth=0.3, zorder=5)

    add_scalebar_and_north(ax, ny_c, nx_c,
                            place_scale='bottom-left',
                            place_north='top-right',
                            color='black')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_color('0.4')
    ax.set_title(r'Zonal mean fixation field $\bar{L}$  ($n = 20$ K-means zones)',
                  fontsize=12)

    plt.subplots_adjust(left=0.02, right=0.9, top=0.94, bottom=0.03)
    cbar_ax = fig.add_axes([0.905, 0.15, 0.014, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'Zonal mean $\bar{L}$', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    state_path = "state_cache.pkl"
    if not Path(state_path).exists():
        print(f"ERROR: {state_path} not found. Run build_state.py first.")
        sys.exit(1)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    print(f"State loaded: {state.n_sites} sites, "
          f"{len(state.attractors)} attractors")

    print("\n=== Figure 3: environmental affinity ===")
    make_fig3(state, "v47_E.png")

    print("\n=== Figure 4: deterministic vs canonical ===")
    make_fig4(state, "figure4_twopanel.png")

    print("\n=== Figure 5: zonal Lbar ===")
    make_fig5(state, "vz1_zones_L_mean.png")

    print("\nAll figures done. Copy them into your figures folder.")


if __name__ == "__main__":
    main()
