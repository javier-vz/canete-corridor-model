#!/usr/bin/env python3
"""
run_ablation_weights.py

Weight-sensitivity ablation (response to JCAA reviewer 2 questions
about how weights a, b were decided and how sensitive results are).

Design:
  - Fix the stochastic parameters at the canonical values
    (eps = 0.15, kappa = 0.03)
  - Vary each component of the affinity weight vector a (a_h, a_p,
    a_w, a_t) by +/-20% of its nominal value, redistributing the
    difference across the other three components so the sum remains 1
  - Do the same for the cost weight vector b (b_h, b_p, b_w, b_t)
  - Baseline: nominal weights
  - Total: 1 baseline + 8 (perturbed a) + 8 (perturbed b) = 17 configs
  - 3 seeds per config = 51 runs, ~7 min each = ~6 hours on 7 workers

For each run, we compute AUC and, if the state pickle has the
attractor+site machinery available, we also compute the zonal
Spearman correlation at n=20 zones (matching Table 4 in the paper).

Outputs:
    ablation_weights_jobs.txt    manifest
    ablation_weights_summary.csv results per config (mean/std over seeds)
    ablation_weights_summary.txt human-readable report

Usage:
    python run_ablation_weights.py               # runs all 51 jobs sequentially with multiprocessing
    python run_ablation_weights.py --workers 7   # explicit worker count
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# --- weight configs ---
NOMINAL_A = (0.10, 0.20, 0.50, 0.20)   # h, p, w, t
NOMINAL_B = (0.18, 0.36, 0.26, 0.20)
PCT = 0.20                              # +/-20%
N_SEEDS = 3
SEED_BASE = 3000
EPS = 0.15
KAP = 0.03
OUT_DIR = Path("results_ablation_weights")


def perturb_weight_vector(w, index, delta_frac):
    """Change w[index] by delta_frac*w[index], redistribute
    the difference across the other three components proportionally.
    Returns new tuple with sum = 1 (up to numerical precision)."""
    w = np.array(w, dtype=float)
    old = w[index]
    new = old * (1.0 + delta_frac)
    diff = new - old
    others = [i for i in range(len(w)) if i != index]
    # Distribute -diff across the other three, proportional to their weights
    total_others = w[others].sum()
    if total_others <= 0:
        raise ValueError("Cannot redistribute; other weights sum to 0")
    for i in others:
        w[i] -= diff * (w[i] / total_others)
    w[index] = new
    if abs(w.sum() - 1.0) > 1e-8:
        raise RuntimeError(f"Weights sum broke: {w.sum()}")
    return tuple(w)


def build_all_configs():
    """Return list of dicts: {config_id, a_vec, b_vec}."""
    configs = [{"config_id": "baseline",
                "a_vec": tuple(NOMINAL_A),
                "b_vec": tuple(NOMINAL_B)}]

    a_names = ["ah", "ap", "aw", "at"]
    for i, name in enumerate(a_names):
        for sign, tag in [(+1, "up"), (-1, "dn")]:
            a_new = perturb_weight_vector(NOMINAL_A, i, sign * PCT)
            configs.append({
                "config_id": f"a_{name}_{tag}",
                "a_vec": a_new,
                "b_vec": tuple(NOMINAL_B),
            })

    b_names = ["bh", "bp", "bw", "bt"]
    for i, name in enumerate(b_names):
        for sign, tag in [(+1, "up"), (-1, "dn")]:
            b_new = perturb_weight_vector(NOMINAL_B, i, sign * PCT)
            configs.append({
                "config_id": f"b_{name}_{tag}",
                "a_vec": tuple(NOMINAL_A),
                "b_vec": b_new,
            })

    return configs


# ------------------------------------------------------------
# Worker functions
# ------------------------------------------------------------

def _init_worker(csv_path):
    """Each worker: import the kernel and load the raw CSV once.
    We build the state per-config (because weights change E and c)."""
    global _WORKER_CSV
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    _WORKER_CSV = csv_path


def _do_job(job):
    """One job: (config_id, a_vec, b_vec, seed, out_path)."""
    from ca_kernel import build_state, run_one
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr
    import numpy as np

    config_id, a_vec, b_vec, seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))
    try:
        t0 = time.time()
        # Build state with these specific weights
        state = build_state(_WORKER_CSV,
                             a_weights=a_vec, b_weights=b_vec)
        # Run the CA
        res = run_one(state, eps=EPS, kappa=KAP, seed=seed)

        # ADD zonal Spearman rho at n=20 (mirrors §4.4 of the paper)
        valid = state.valid_mask
        # Coordinates of valid cells
        yy, xx = np.where(valid)
        h_at = state.h_grid[valid]
        p_at = state.p_grid[valid]
        # Feature matrix for K-means (x, y, h, p) standardised
        X = np.column_stack([xx, yy, h_at, p_at]).astype(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        km = KMeans(n_clusters=20, n_init=10, random_state=0)
        labels = km.fit_predict(X)

        # Zonal mean of L (we don't have L directly; recompute)
        from ca_kernel import build_fixation_field, apply_perceptual_noise, \
            apply_congestion_update, k_simple_paths_weighted, path_cost
        import networkx as nx
        from copy import deepcopy

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

        # Zonal means and site counts
        site_at_valid = state.site_mask[valid].astype(int)
        L_mean = np.array([L_at_valid[labels == z].mean()
                            for z in range(20)])
        site_count = np.array([site_at_valid[labels == z].sum()
                                for z in range(20)])
        rho, p_rho = spearmanr(L_mean, site_count)

        result = {
            "config_id": config_id,
            "a_vec": list(a_vec),
            "b_vec": list(b_vec),
            "seed": seed,
            "eps": EPS,
            "kappa": KAP,
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
        return ("done", str(out), res["auc"], float(rho))
    except Exception as e:
        return ("error", str(out), repr(e))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="grilla_con_W_hidrica.csv")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    # Build job list
    configs = build_all_configs()
    print(f"Configs: {len(configs)}")
    print(f"Seeds per config: {N_SEEDS}")

    jobs = []
    for cfg in configs:
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            out = OUT_DIR / f"{cfg['config_id']}_s{seed}.json"
            jobs.append((cfg["config_id"],
                          cfg["a_vec"], cfg["b_vec"],
                          seed, str(out)))

    todo = [j for j in jobs if not Path(j[4]).exists()]
    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

    print(f"Workers: {n_workers}")
    print(f"Total jobs: {len(jobs)}   To do: {len(todo)}")
    print(f"Estimated wall time: ~{len(todo) * 7 / n_workers:.1f} minutes "
          f"( ~{len(todo) * 7 / n_workers / 60:.1f} hours)")

    if len(todo) == 0:
        print("Nothing to do; running analysis only.")
    else:
        t0 = time.time()
        completed = 0
        with mp.Pool(processes=n_workers,
                     initializer=_init_worker,
                     initargs=(str(csv_path),)) as pool:
            try:
                for res in pool.imap_unordered(_do_job, todo, chunksize=1):
                    completed += 1
                    if res[0] == "done":
                        _, out, auc, rho = res
                        el = time.time() - t0
                        eta = el / completed * (len(todo) - completed)
                        print(f"[{completed:3d}/{len(todo)}] "
                              f"AUC={auc:.4f}  rho={rho:+.3f}  "
                              f"({el/60:.1f}m elapsed, "
                              f"eta {eta/60:.1f}m)")
                    elif res[0] == "error":
                        print(f"[{completed:3d}/{len(todo)}] ERROR: {res[2]}")
            except KeyboardInterrupt:
                print("\nInterrupted -- results so far saved.")
                pool.terminate()
                pool.join()
                sys.exit(130)
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    # --- Analyze ---
    import pandas as pd
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        print("No results yet.")
        return
    df = pd.DataFrame(rows)

    # Aggregate per config_id
    summary = df.groupby("config_id").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rho_mean=("rho_n20", "mean"), rho_std=("rho_n20", "std"),
        m2_mean=("m2", "mean"), m2_std=("m2", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv("ablation_weights_summary.csv", index=False)
    print("\nWrote ablation_weights_summary.csv")

    # Human summary
    lines = ["# Weight-ablation summary",
             f"# Perturbation: +/-{PCT*100:.0f}% per component, "
             "redistributing across the others",
             ""]
    lines.append(f"{'Config':<20s} {'AUC':>16s} {'rho (n=20)':>16s} {'m2':>14s}")
    lines.append("-" * 70)
    for _, row in summary.iterrows():
        lines.append(
            f"{row['config_id']:<20s} "
            f"{row['auc_mean']:.4f} ± {row['auc_std']:.4f}   "
            f"{row['rho_mean']:+.3f} ± {row['rho_std']:.3f}   "
            f"{row['m2_mean']:.3f} ± {row['m2_std']:.3f}"
        )

    # Ranges
    lines.append("")
    lines.append("=== RANGES ACROSS CONFIGS ===")
    lines.append(f"AUC : [{summary['auc_mean'].min():.4f}, "
                 f"{summary['auc_mean'].max():.4f}]  "
                 f"range = {summary['auc_mean'].max()-summary['auc_mean'].min():.4f}")
    lines.append(f"rho : [{summary['rho_mean'].min():+.3f}, "
                 f"{summary['rho_mean'].max():+.3f}]  "
                 f"range = {summary['rho_mean'].max()-summary['rho_mean'].min():.3f}")

    txt = "\n".join(lines)
    with open("ablation_weights_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("\nWrote ablation_weights_summary.txt")


if __name__ == "__main__":
    mp.freeze_support()
    main()
