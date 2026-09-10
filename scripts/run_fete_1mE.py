#!/usr/bin/env python3
"""
run_fete_1mE.py

FETE baseline variant using (1 - E) as cost surface, rather than
the standard cost c. This is a strict test of whether the affinity
field E alone (inverted to serve as routing cost) produces comparable
or superior corridor structure to the composite cost surface c.

Design:
    - n_pairs = 86 (matched to CA canonical)
    - 30 seeds (matched to standard FETE from run_fete_local.py)
    - Same random-endpoint sampling per seed
    - Only difference from the standard FETE: cost surface = 1 - E

Estimated: 30 runs, ~7-8 min each = ~4 hours on 7 workers (or ~30 min
if all fit; FETE is faster than CA because it's single-path per seed).

Outputs:
    results_fete_1mE/{fete_1mE_n086_s0000.json ...}
    fete_1mE_summary.txt

Usage:
    python run_fete_1mE.py
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

N_PAIRS = 86
N_SEEDS = 30
# SEED_BASE MUST match run_fete_local.py (which uses 2000-2029) so that
# each seed produces IDENTICAL random endpoint pairs across both variants.
# This allows seed-by-seed paired comparison of FETE(c) vs FETE(1-E), not
# just aggregate mean comparison. Previously used 6000-6029 which
# generated a completely different endpoint set per seed.
SEED_BASE = 2000
OUT_DIR = Path("results_fete_1mE_matched")


def _init_worker(csv_path):
    global _WORKER_CSV, _WORKER_STATE
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ca_kernel import build_state
    _WORKER_CSV = csv_path
    _WORKER_STATE = build_state(csv_path)


def _do_job(job):
    from ca_kernel import run_fete
    seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))
    try:
        t0 = time.time()
        # Build 1-E cost surface
        cost_1mE = 1.0 - _WORKER_STATE.E_grid
        res = run_fete(_WORKER_STATE, n_random_pairs=N_PAIRS, seed=seed,
                        cost_surface=cost_1mE)
        res["method"] = "fete_1mE"
        res["cost_surface"] = "1-E"
        res["runtime_s"] = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        return ("done", str(out), res["auc"], res["lcc_size"])
    except Exception as e:
        return ("error", str(out), repr(e))


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

    OUT_DIR.mkdir(exist_ok=True)

    jobs = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        out = OUT_DIR / f"fete_1mE_n{N_PAIRS:03d}_s{seed:04d}.json"
        jobs.append((seed, str(out)))

    todo = [j for j in jobs if not Path(j[1]).exists()]
    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

    print(f"FETE (1-E cost): {N_PAIRS} pairs x {N_SEEDS} seeds "
          f"= {len(jobs)} runs, {len(todo)} to do")
    print(f"Workers: {n_workers}")
    print(f"Estimated: ~{len(todo) * 7 / n_workers:.0f} min "
          f"(FETE is faster than CA)\n")

    if len(todo) > 0:
        t0 = time.time()
        completed = 0
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                     initargs=(str(csv_path),)) as pool:
            try:
                for res in pool.imap_unordered(_do_job, todo, chunksize=1):
                    completed += 1
                    if res[0] == "done":
                        _, out, auc, lcc = res
                        print(f"[{completed:2d}/{len(todo)}] "
                              f"AUC={auc:.4f} LCC={lcc}  "
                              f"({Path(out).name})")
                    elif res[0] == "error":
                        print(f"[{completed:2d}/{len(todo)}] ERROR: {res[2]}")
            except KeyboardInterrupt:
                print("\nInterrupted. Safe to resume.")
                pool.terminate(); pool.join(); sys.exit(130)
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    # Aggregate
    import pandas as pd
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        with open(f) as fh: rows.append(json.load(fh))
    if not rows:
        print("No results yet."); return

    df = pd.DataFrame(rows)
    print(f"\nAggregated {len(df)} runs")
    lines = ["# FETE (1-E cost surface) summary",
             "# Compare against standard FETE (c cost) and CA canonical", ""]
    lines.append(f"{'metric':<10s} {'mean':>10s} {'std':>10s}")
    lines.append("-" * 40)
    for m in ["auc", "lcc_size", "m1", "m2"]:
        lines.append(f"{m:<10s} {df[m].mean():10.4f} {df[m].std():10.4f}")
    lines.append("")
    lines.append("For reference (from previous runs):")
    lines.append("  Standard FETE (c cost, n=86):  AUC = 0.810 +/- 0.008, "
                 "LCC = 146 +/- 20, m2 = 0.27 +/- 0.05")
    lines.append("  CA canonical (eps=0.15, kap=0.03): AUC = 0.816 +/- 0.002, "
                 "LCC = 138 +/- 23, m2 = 0.348 +/- 0.061")

    txt = "\n".join(lines)
    with open("fete_1mE_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    mp.freeze_support()
    main()
