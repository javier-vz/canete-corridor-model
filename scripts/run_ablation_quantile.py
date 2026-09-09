#!/usr/bin/env python3
"""
run_ablation_quantile.py

Sensitivity analysis for the attractor extraction quantile tau_E,
responding to reviewer requests about the sensitivity of the model
to the 0.92 quantile choice.

Design:
    - Fix stochastic parameters at canonical values (eps=0.15, kappa=0.03)
    - Vary tau_E in {0.90, 0.92, 0.95, 0.975}
    - 3 seeds per config
    - Total: 4 configs * 3 seeds = 12 runs, ~7 min each = ~1.5 hours
      on 7 workers (parallelizes to 6 minutes if all fit).

IMPORTANT: unlike the weight ablation, this changes the SET OF
ATTRACTORS, which changes the number of pairs and the topology of
the corridor system. This is desired: we want to see how the model
responds to a genuinely different attractor configuration, not force
24 attractors artificially.

Outputs:
    results_quantile/{quantile_q0.900_s3000.json ...}
    ablation_quantile_summary.csv
    ablation_quantile_summary.txt

Usage:
    python run_ablation_quantile.py
    python run_ablation_quantile.py --workers 7
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Configuration
QUANTILES = [0.90, 0.92, 0.95, 0.975]
N_SEEDS = 3
SEED_BASE = 4000
EPS = 0.15
KAP = 0.03
OUT_DIR = Path("results_quantile")


def _init_worker(csv_path):
    global _WORKER_CSV
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    _WORKER_CSV = csv_path


def _do_job(job):
    from ca_kernel import build_state, run_one, build_fixation_field
    from ca_kernel import apply_perceptual_noise, apply_congestion_update
    from ca_kernel import k_simple_paths_weighted, path_cost
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr
    import networkx as nx
    from copy import deepcopy

    quantile, seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))

    try:
        t0 = time.time()
        # Build state with this quantile (default weights)
        state = build_state(_WORKER_CSV, peak_quantile=quantile)
        # Run the CA
        res = run_one(state, eps=EPS, kappa=KAP, seed=seed)

        # Compute zonal Spearman rho at n=20
        valid = state.valid_mask
        yy, xx = np.where(valid)
        h_at = state.h_grid[valid]
        p_at = state.p_grid[valid]
        X = np.column_stack([xx, yy, h_at, p_at]).astype(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        km = KMeans(n_clusters=20, n_init=10, random_state=0)
        labels = km.fit_predict(X)

        # Re-run to get F (deterministic given seed)
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
            apply_perceptual_noise(G, EPS, rng)
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
            apply_congestion_update(G, visited, kappa=KAP, cap=2.0)
        F_norm = traffic / (traffic.sum() + 1e-12)
        L_field, _, _ = build_fixation_field(F_norm, state.T_grid, valid)
        L_at_valid = L_field[valid]

        site_at_valid = state.site_mask[valid].astype(int)
        L_mean = np.array([L_at_valid[labels == z].mean()
                            for z in range(20)])
        site_count = np.array([site_at_valid[labels == z].sum()
                                for z in range(20)])
        rho, p_rho = spearmanr(L_mean, site_count)

        result = {
            "quantile": quantile,
            "seed": seed,
            "eps": EPS,
            "kappa": KAP,
            "n_attractors": len(state.attractors),
            "n_pairs": len(state.pair_candidates),
            "auc": res["auc"],
            "m1": res["m1"],
            "m2": res["m2"],
            "m3": res["m3"],
            "lcc_size": res["lcc_size"],
            "rho_n20": float(rho),
            "p_rho_n20": float(p_rho),
            "runtime_s": time.time() - t0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        return ("done", str(out),
                len(state.attractors), len(state.pair_candidates),
                res["auc"], float(rho))
    except Exception as e:
        return ("error", str(out), repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/grilla_con_W_hidrica.csv")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try common relative paths
        for cand in ["grilla_con_W_hidrica.csv",
                     "../data/grilla_con_W_hidrica.csv"]:
            if Path(cand).exists():
                csv_path = Path(cand)
                break
        else:
            print(f"ERROR: CSV not found. Tried: {args.csv}, "
                  "grilla_con_W_hidrica.csv, ../data/grilla_con_W_hidrica.csv")
            sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    jobs = []
    for q in QUANTILES:
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            out = OUT_DIR / f"quantile_q{q:.3f}_s{seed}.json"
            jobs.append((q, seed, str(out)))

    todo = [j for j in jobs if not Path(j[2]).exists()]
    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

    print(f"Configs: {len(QUANTILES)} quantiles x {N_SEEDS} seeds "
          f"= {len(jobs)} jobs, {len(todo)} to do")
    print(f"Workers: {n_workers}")
    print(f"Estimated: ~{len(todo) * 7 / n_workers:.0f} min\n")

    if len(todo) > 0:
        t0 = time.time()
        completed = 0
        with mp.Pool(processes=n_workers,
                     initializer=_init_worker,
                     initargs=(str(csv_path),)) as pool:
            try:
                for res in pool.imap_unordered(_do_job, todo, chunksize=1):
                    completed += 1
                    if res[0] == "done":
                        _, out, n_att, n_pairs, auc, rho = res
                        print(f"[{completed:2d}/{len(todo)}] "
                              f"n_att={n_att:2d} n_pairs={n_pairs:3d} "
                              f"AUC={auc:.4f} rho={rho:+.3f}  ({out})")
                    elif res[0] == "error":
                        print(f"[{completed:2d}/{len(todo)}] ERROR: {res[2]}")
            except KeyboardInterrupt:
                print("\nInterrupted. Safe to resume by re-running.")
                pool.terminate(); pool.join()
                sys.exit(130)
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    # Aggregate
    import pandas as pd
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        print("No results yet.")
        return

    df = pd.DataFrame(rows)
    summary = df.groupby("quantile").agg(
        n_attractors=("n_attractors", "first"),
        n_pairs=("n_pairs", "first"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rho_mean=("rho_n20", "mean"), rho_std=("rho_n20", "std"),
        lcc_mean=("lcc_size", "mean"), lcc_std=("lcc_size", "std"),
        m2_mean=("m2", "mean"), m2_std=("m2", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv("ablation_quantile_summary.csv", index=False)
    print("\nWrote ablation_quantile_summary.csv")

    lines = ["# Quantile ablation summary",
             "# Response to reviewer question on sensitivity of tau_E", ""]
    lines.append(f"{'q':>7s} {'n_att':>7s} {'n_pairs':>9s} "
                 f"{'AUC':>16s} {'rho (n=20)':>16s} {'LCC':>14s}")
    lines.append("-" * 78)
    for _, r in summary.iterrows():
        lines.append(
            f"{r['quantile']:7.3f} {int(r['n_attractors']):7d} "
            f"{int(r['n_pairs']):9d} "
            f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}   "
            f"{r['rho_mean']:+.3f} +/- {r['rho_std']:.3f}   "
            f"{r['lcc_mean']:6.1f} +/- {r['lcc_std']:5.1f}"
        )
    lines.append("")
    lines.append("=== RANGES ACROSS QUANTILES ===")
    lines.append(f"AUC: [{summary['auc_mean'].min():.4f}, "
                 f"{summary['auc_mean'].max():.4f}]  "
                 f"range = {summary['auc_mean'].max()-summary['auc_mean'].min():.4f}")
    lines.append(f"rho: [{summary['rho_mean'].min():+.3f}, "
                 f"{summary['rho_mean'].max():+.3f}]  "
                 f"range = {summary['rho_mean'].max()-summary['rho_mean'].min():.3f}")

    txt = "\n".join(lines)
    with open("ablation_quantile_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("\nWrote ablation_quantile_summary.txt")


if __name__ == "__main__":
    mp.freeze_support()
    main()
