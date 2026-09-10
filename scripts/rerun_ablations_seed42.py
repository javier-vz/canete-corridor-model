#!/usr/bin/env python3
"""
rerun_ablations_seed42.py

Re-runs the three ablation experiments with K-means random_state=42
(matching the paper's original K-means partition, which gives rho=0.638
for the canonical CA), instead of the random_state=0 used in the
first pass.

This resolves the discrepancy noted in the reviewer feedback: the
paper's zonal Spearman correlations use seed=42; earlier ablation
scripts used seed=0, giving a different partition and rho.

Design:
    Same three ablations as before, same seeds, same everything --
    only change is random_state=42 in KMeans.

    Total: (17 + 4 + 7) configs * 3 seeds = 84 runs, ~7 min each
    = ~10 hours on 7 workers. Run overnight.

Reads existing JSONs if present (idempotent), only re-runs configs
that don't have a seed42 output.

Outputs:
    results_ablation_weights_s42/
    results_quantile_s42/
    results_T_s42/
    ablation_*_s42_summary.txt (three files)

Usage:
    python rerun_ablations_seed42.py
    python rerun_ablations_seed42.py --workers 7
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np

# Nominal
NOMINAL_A = (0.10, 0.20, 0.50, 0.20)
NOMINAL_B = (0.18, 0.36, 0.26, 0.20)
PCT = 0.20
EPS = 0.15
KAP = 0.03
KMEANS_SEED = 42   # <<< KEY CHANGE: match the paper

N_SEEDS = 3
SEED_BASE = 3000  # reuse the same seeds as previous runs for consistency


def perturb_weight_vector(w, index, delta_frac):
    w = np.array(w, dtype=float)
    old = w[index]
    new = old * (1.0 + delta_frac)
    diff = new - old
    others = [i for i in range(len(w)) if i != index]
    total_others = w[others].sum()
    for i in others:
        w[i] -= diff * (w[i] / total_others)
    w[index] = new
    return tuple(w)


def build_weight_configs():
    configs = [("baseline", tuple(NOMINAL_A), tuple(NOMINAL_B))]
    a_names = ["ah", "ap", "aw", "at"]
    for i, name in enumerate(a_names):
        for sign, tag in [(+1, "up"), (-1, "dn")]:
            a_new = perturb_weight_vector(NOMINAL_A, i, sign * PCT)
            configs.append((f"a_{name}_{tag}", a_new, tuple(NOMINAL_B)))
    b_names = ["bh", "bp", "bw", "bt"]
    for i, name in enumerate(b_names):
        for sign, tag in [(+1, "up"), (-1, "dn")]:
            b_new = perturb_weight_vector(NOMINAL_B, i, sign * PCT)
            configs.append((f"b_{name}_{tag}", tuple(NOMINAL_A), b_new))
    return configs


T_CONFIGS = [
    ("baseline", 0.65, 0.20, 5),
    ("r_low",    0.50, 0.20, 5),
    ("r_high",   0.80, 0.20, 5),
    ("s_zero",   0.65, 0.00, 5),
    ("s_high",   0.65, 0.40, 5),
    ("w_small",  0.65, 0.20, 3),
    ("w_large",  0.65, 0.20, 7),
]

QUANTILES = [0.90, 0.92, 0.95, 0.975]


def _init_worker(csv_path):
    global _WORKER_CSV
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    _WORKER_CSV = csv_path


def _compute_rho(state, res_seed, kmeans_random_state):
    """Compute zonal Spearman rho, given a completed run's seed
    (deterministic given seed). Rebuilds L from scratch."""
    from ca_kernel import (run_one, apply_perceptual_noise,
                            apply_congestion_update,
                            k_simple_paths_weighted, path_cost,
                            build_fixation_field)
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr
    import networkx as nx
    from copy import deepcopy

    # Use return_fields to get L directly (avoids re-implementing the loop)
    r = run_one(state, eps=EPS, kappa=KAP, seed=res_seed,
                 return_fields=True)
    L_field = r["L"]

    valid = state.valid_mask
    yy, xx = np.where(valid)
    h_at = state.h_grid[valid]
    p_at = state.p_grid[valid]
    X = np.column_stack([xx, yy, h_at, p_at]).astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    km = KMeans(n_clusters=20, n_init=10, random_state=kmeans_random_state)
    labels = km.fit_predict(X)

    L_at_valid = L_field[valid]
    site_at_valid = state.site_mask[valid].astype(int)
    L_mean = np.array([L_at_valid[labels == z].mean() for z in range(20)])
    site_count = np.array([site_at_valid[labels == z].sum() for z in range(20)])
    rho, p_rho = spearmanr(L_mean, site_count)
    return float(rho), float(p_rho), r


def _do_job_weights(job):
    from ca_kernel import build_state
    config_id, a_vec, b_vec, seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))
    try:
        t0 = time.time()
        state = build_state(_WORKER_CSV, a_weights=a_vec, b_weights=b_vec)
        rho, p_rho, res = _compute_rho(state, seed, KMEANS_SEED)
        result = {
            "config_id": config_id, "a_vec": list(a_vec), "b_vec": list(b_vec),
            "seed": seed, "eps": EPS, "kappa": KAP,
            "kmeans_random_state": KMEANS_SEED,
            "auc": res["auc"], "m1": res["m1"], "m2": res["m2"], "m3": res["m3"],
            "lcc_size": res["lcc_size"],
            "rho_n20": rho, "p_rho_n20": p_rho,
            "runtime_s": time.time() - t0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f: json.dump(result, f, indent=2)
        return ("done", str(out), res["auc"], rho)
    except Exception as e:
        return ("error", str(out), repr(e))


def _do_job_T(job):
    from ca_kernel import build_state
    config_id, r, s, w, seed, out_path = job
    out = Path(out_path)
    if out.exists(): return ("skip", str(out))
    try:
        t0 = time.time()
        state = build_state(_WORKER_CSV, T_r=r, T_s=s, T_window=w)
        rho, p_rho, res = _compute_rho(state, seed, KMEANS_SEED)
        result = {
            "config_id": config_id, "T_r": r, "T_s": s, "T_window": w,
            "seed": seed, "eps": EPS, "kappa": KAP,
            "kmeans_random_state": KMEANS_SEED,
            "auc": res["auc"], "m1": res["m1"], "m2": res["m2"], "m3": res["m3"],
            "lcc_size": res["lcc_size"],
            "n_attractors": len(state.attractors),
            "n_pairs": len(state.pair_candidates),
            "rho_n20": rho, "p_rho_n20": p_rho,
            "runtime_s": time.time() - t0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f: json.dump(result, f, indent=2)
        return ("done", str(out), res["auc"], rho)
    except Exception as e:
        return ("error", str(out), repr(e))


def _do_job_quantile(job):
    from ca_kernel import build_state
    quantile, seed, out_path = job
    out = Path(out_path)
    if out.exists(): return ("skip", str(out))
    try:
        t0 = time.time()
        state = build_state(_WORKER_CSV, peak_quantile=quantile)
        rho, p_rho, res = _compute_rho(state, seed, KMEANS_SEED)
        result = {
            "quantile": quantile, "seed": seed, "eps": EPS, "kappa": KAP,
            "kmeans_random_state": KMEANS_SEED,
            "auc": res["auc"], "m1": res["m1"], "m2": res["m2"], "m3": res["m3"],
            "lcc_size": res["lcc_size"],
            "n_attractors": len(state.attractors),
            "n_pairs": len(state.pair_candidates),
            "rho_n20": rho, "p_rho_n20": p_rho,
            "runtime_s": time.time() - t0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f: json.dump(result, f, indent=2)
        return ("done", str(out), res["auc"], rho)
    except Exception as e:
        return ("error", str(out), repr(e))


def summarize_weights(out_dir):
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(out_dir.glob("*.json"))]
    if not rows: return None
    df = pd.DataFrame(rows)
    s = df.groupby("config_id").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rho_mean=("rho_n20", "mean"), rho_std=("rho_n20", "std"),
        m2_mean=("m2", "mean"), m2_std=("m2", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    s.to_csv("ablation_weights_s42_summary.csv", index=False)
    lines = ["# Weight ablation summary (K-means random_state=42)", ""]
    lines.append(f"{'config':<20s} {'AUC':>16s} {'rho (n=20)':>16s}")
    lines.append("-" * 60)
    for _, r in s.iterrows():
        lines.append(f"{r['config_id']:<20s} "
                     f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}   "
                     f"{r['rho_mean']:+.3f} +/- {r['rho_std']:.3f}")
    lines.append("")
    lines.append(f"AUC range: [{s['auc_mean'].min():.4f}, {s['auc_mean'].max():.4f}]")
    lines.append(f"rho range: [{s['rho_mean'].min():+.3f}, {s['rho_mean'].max():+.3f}]")
    with open("ablation_weights_s42_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def summarize_T(out_dir):
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(out_dir.glob("*.json"))]
    if not rows: return None
    df = pd.DataFrame(rows)
    s = df.groupby("config_id").agg(
        T_r=("T_r", "first"), T_s=("T_s", "first"), T_window=("T_window", "first"),
        n_attractors=("n_attractors", "first"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rho_mean=("rho_n20", "mean"), rho_std=("rho_n20", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    s.to_csv("ablation_T_s42_summary.csv", index=False)
    lines = ["# T-parameter ablation (K-means random_state=42)", ""]
    lines.append(f"{'config':<12s} {'r':>5s} {'s':>5s} {'w':>3s} "
                 f"{'n_att':>6s} {'AUC':>16s} {'rho':>16s}")
    lines.append("-" * 78)
    for _, r in s.iterrows():
        lines.append(
            f"{r['config_id']:<12s} {r['T_r']:5.2f} {r['T_s']:5.2f} "
            f"{int(r['T_window']):3d} {int(r['n_attractors']):6d} "
            f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}   "
            f"{r['rho_mean']:+.3f} +/- {r['rho_std']:.3f}"
        )
    lines.append("")
    lines.append(f"AUC range: [{s['auc_mean'].min():.4f}, {s['auc_mean'].max():.4f}]")
    lines.append(f"rho range: [{s['rho_mean'].min():+.3f}, {s['rho_mean'].max():+.3f}]")
    with open("ablation_T_s42_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def summarize_quantile(out_dir):
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(out_dir.glob("*.json"))]
    if not rows: return None
    df = pd.DataFrame(rows)
    s = df.groupby("quantile").agg(
        n_attractors=("n_attractors", "first"),
        n_pairs=("n_pairs", "first"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        rho_mean=("rho_n20", "mean"), rho_std=("rho_n20", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    s.to_csv("ablation_quantile_s42_summary.csv", index=False)
    lines = ["# Quantile ablation (K-means random_state=42)", ""]
    lines.append(f"{'q':>7s} {'n_att':>7s} {'n_pairs':>9s} "
                 f"{'AUC':>16s} {'rho (n=20)':>16s}")
    lines.append("-" * 66)
    for _, r in s.iterrows():
        lines.append(f"{r['quantile']:7.3f} {int(r['n_attractors']):7d} "
                     f"{int(r['n_pairs']):9d} "
                     f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}   "
                     f"{r['rho_mean']:+.3f} +/- {r['rho_std']:.3f}")
    lines.append("")
    lines.append(f"AUC range: [{s['auc_mean'].min():.4f}, {s['auc_mean'].max():.4f}]")
    lines.append(f"rho range: [{s['rho_mean'].min():+.3f}, {s['rho_mean'].max():+.3f}]")
    with open("ablation_quantile_s42_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/grilla_con_W_hidrica.csv")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        for cand in ["grilla_con_W_hidrica.csv",
                     "../data/grilla_con_W_hidrica.csv"]:
            if Path(cand).exists():
                csv_path = Path(cand); break
        else:
            print(f"ERROR: CSV not found."); sys.exit(1)

    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

    # -------- WEIGHTS --------
    OUT_W = Path("results_ablation_weights_s42"); OUT_W.mkdir(exist_ok=True)
    jobs_w = []
    for cid, a, b in build_weight_configs():
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            out = OUT_W / f"{cid}_s{seed}.json"
            jobs_w.append((cid, a, b, seed, str(out)))

    # -------- T --------
    OUT_T = Path("results_T_s42"); OUT_T.mkdir(exist_ok=True)
    jobs_T = []
    for cid, r, s, w in T_CONFIGS:
        for i in range(N_SEEDS):
            seed = 5000 + i
            out = OUT_T / f"T_{cid}_r{r:.3f}_s{s:.3f}_w{w}_s{seed}.json"
            jobs_T.append((cid, r, s, w, seed, str(out)))

    # -------- QUANTILE --------
    OUT_Q = Path("results_quantile_s42"); OUT_Q.mkdir(exist_ok=True)
    jobs_Q = []
    for q in QUANTILES:
        for i in range(N_SEEDS):
            seed = 4000 + i
            out = OUT_Q / f"quantile_q{q:.3f}_s{seed}.json"
            jobs_Q.append((q, seed, str(out)))

    todo_w = [j for j in jobs_w if not Path(j[4]).exists()]
    todo_T = [j for j in jobs_T if not Path(j[5]).exists()]
    todo_Q = [j for j in jobs_Q if not Path(j[2]).exists()]
    total = len(todo_w) + len(todo_T) + len(todo_Q)
    print(f"To do: {len(todo_w)} weight, {len(todo_T)} T, {len(todo_Q)} quantile "
          f"= {total} runs")
    print(f"Workers: {n_workers}")
    print(f"Estimated: ~{total * 7 / n_workers:.0f} min "
          f"(~{total * 7 / n_workers / 60:.1f} h)\n")

    for label, jobs, worker_fn in [
        ("weights",  todo_w, _do_job_weights),
        ("T",        todo_T, _do_job_T),
        ("quantile", todo_Q, _do_job_quantile),
    ]:
        if not jobs: continue
        print(f"=== {label.upper()} ({len(jobs)} jobs) ===")
        t0 = time.time()
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                     initargs=(str(csv_path),)) as pool:
            try:
                for i, res in enumerate(pool.imap_unordered(worker_fn, jobs,
                                                             chunksize=1), 1):
                    if res[0] == "done":
                        _, out, auc, rho = res
                        print(f"[{i:3d}/{len(jobs)}] AUC={auc:.4f} rho={rho:+.3f}")
                    elif res[0] == "error":
                        print(f"[{i:3d}/{len(jobs)}] ERROR: {res[2]}")
            except KeyboardInterrupt:
                print("Interrupted."); pool.terminate(); pool.join(); sys.exit(130)
        print(f"  done in {(time.time()-t0)/60:.1f} min\n")

    # Summarize
    print("=== SUMMARIES ===")
    for name, fn, d in [("Weights",  summarize_weights,  OUT_W),
                         ("T",        summarize_T,        OUT_T),
                         ("Quantile", summarize_quantile, OUT_Q)]:
        txt = fn(d)
        if txt:
            print(f"\n--- {name} ---")
            print(txt)


if __name__ == "__main__":
    mp.freeze_support()
    main()
