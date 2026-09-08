#!/usr/bin/env python3
"""
analyze_ks_test.py
Kolmogorov-Smirnov comparison of L (and E, F, T) at site cells versus
background cells. This is the "standard predictive-modelling assessment"
requested by JCAA reviewer 1:

  "the authors could compare the E values (formula 3) and fixation
   field values L (formula 13) of the grid with those of the sites
   (or a subset thereof, omitting some of the sites in the main
   cluster) by a Kolmogorov Smirnov test..."

We run KS on:
  - E field (built once from state)
  - L field for the CA canonical (eps=0.15, kappa=0.03)
  - and, if requested, on a leave-one-cluster-out variant
    (drop the top archaeological cluster, redo KS)

This requires the *fields* (not just observables) so we run one CA
inline here and reuse the state.

Usage:
    python analyze_ks_test.py --state state_cache.pkl \\
        --eps 0.15 --kappa 0.03 --seed 42
"""
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from scipy.ndimage import binary_dilation, label as scilabel
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9


def ks_report(field, site_mask, valid_mask, name, out_lines):
    pos = field[site_mask]
    neg = field[valid_mask & ~site_mask]
    stat, p = ks_2samp(pos, neg, alternative='two-sided')
    mean_pos = float(pos.mean())
    mean_neg = float(neg.mean())
    out_lines.append(f"  {name}:")
    out_lines.append(f"    KS statistic D = {stat:.4f}")
    out_lines.append(f"    p-value        = {p:.3e}")
    out_lines.append(f"    <field | site> = {mean_pos:.4f}   (n={len(pos)})")
    out_lines.append(f"    <field | bg>   = {mean_neg:.4f}   (n={len(neg)})")
    out_lines.append(f"    contrast       = {mean_pos - mean_neg:+.4f}")
    return {'name': name, 'D': stat, 'p': p,
            'mean_pos': mean_pos, 'mean_neg': mean_neg}


def dominant_cluster_mask(site_mask, dilation_r=3):
    """Return a mask of the 'dominant' archaeological cluster.
    Sites are dilated by a small radius to group nearby sites into a
    single cluster, then the largest connected component is returned
    as the dominant cluster.
    """
    struct = np.ones((2 * dilation_r + 1, 2 * dilation_r + 1))
    dilated = binary_dilation(site_mask, structure=struct)
    labeled, n = scilabel(dilated)
    if n == 0:
        return np.zeros_like(site_mask)
    sizes = np.bincount(labeled.ravel())[1:]
    dom_label = int(sizes.argmax()) + 1
    dom_mask = (labeled == dom_label) & site_mask
    return dom_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state_cache.pkl")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ca_kernel import (
        run_one, build_fixation_field, apply_perceptual_noise,
        apply_congestion_update, k_simple_paths_weighted, path_cost
    )
    import networkx as nx
    from copy import deepcopy

    with open(args.state, "rb") as f:
        state = pickle.load(f)

    # We need the full L field, so we re-run the CA inline
    G = deepcopy(state.G_template)
    rng = np.random.default_rng(args.seed)

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
        apply_perceptual_noise(G, args.eps, rng)
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
        apply_congestion_update(G, visited, kappa=args.kappa, cap=2.0)

    F_norm = traffic / (traffic.sum() + 1e-12)
    L_field, _, _ = build_fixation_field(F_norm, state.T_grid,
                                          state.valid_mask)

    lines = []
    lines.append("# Kolmogorov-Smirnov diagnostic")
    lines.append("# Requested by JCAA reviewer 1")
    lines.append(f"# CA config: eps={args.eps}, kappa={args.kappa}, "
                 f"seed={args.seed}")
    lines.append("")
    lines.append("## Full-site KS (all 84 sites vs background)")
    r1 = ks_report(state.E_grid, state.site_mask, state.valid_mask,
                    "E (affinity)", lines)
    r2 = ks_report(state.T_grid, state.site_mask, state.valid_mask,
                    "T (terrace)", lines)
    r3 = ks_report(state.w_grid, state.site_mask, state.valid_mask,
                    "W (hydrology)", lines)
    r4 = ks_report(L_field, state.site_mask, state.valid_mask,
                    "L (fixation)", lines)
    r5 = ks_report(F_norm, state.site_mask, state.valid_mask,
                    "F (traffic)", lines)

    # Leave-dominant-cluster-out KS: drops the top archaeological cluster
    dom_mask = dominant_cluster_mask(state.site_mask, dilation_r=3)
    n_dom = int(dom_mask.sum())
    residual_mask = state.site_mask & ~dom_mask
    n_res = int(residual_mask.sum())
    lines.append("")
    lines.append(f"## Leave-dominant-cluster-out KS  "
                 f"({n_dom} dominant sites removed, {n_res} residual)")
    ks_report(L_field, residual_mask, state.valid_mask,
              "L (fixation, residual sites)", lines)
    ks_report(state.E_grid, residual_mask, state.valid_mask,
              "E (affinity, residual sites)", lines)

    # Save
    with open("ks_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote ks_report.txt")

    # Also make a plot: CDFs of L on sites vs background
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, field, name in [(axes[0], state.E_grid, "E (environmental affinity)"),
                             (axes[1], L_field,     "L (fixation field)")]:
        pos = np.sort(field[state.site_mask])
        neg = np.sort(field[state.valid_mask & ~state.site_mask])
        ax.plot(pos, np.linspace(0, 1, len(pos)),
                 lw=2, color='crimson', label=f'Sites (n={len(pos)})')
        ax.plot(neg, np.linspace(0, 1, len(neg)),
                 lw=1, color='steelblue', label=f'Background (n={len(neg)})')
        ax.set_xlabel(name)
        ax.set_ylabel('CDF')
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ks_cdfs.png", dpi=150)
    print("Wrote ks_cdfs.png")


if __name__ == "__main__":
    main()
