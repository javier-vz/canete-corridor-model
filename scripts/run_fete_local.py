#!/usr/bin/env python3
"""
run_fete_local.py

Local Windows/Anaconda runner for the FETE baseline experiment.
Same design as run_phase1_local.py but for FETE.

150 jobs, each ~2s: total <1 minute on 8 cores.

Usage:
    python run_fete_local.py
    python run_fete_local.py --workers 6
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

JOBS_FILE = "fete_jobs.txt"
STATE_FILE = "state_cache.pkl"


def _init_worker(state_path):
    global _WORKER_STATE, _WORKER_KERNEL
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ca_kernel import run_fete as _run_fete
    _WORKER_KERNEL = _run_fete
    with open(state_path, "rb") as f:
        _WORKER_STATE = pickle.load(f)


def _do_job(job):
    n_pairs, seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))
    try:
        t0 = time.time()
        res = _WORKER_KERNEL(_WORKER_STATE, n_random_pairs=n_pairs,
                              seed=seed)
        res["runtime_s"] = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        return ("done", str(out), res["auc"], res["m2"], n_pairs)
    except Exception as e:
        return ("error", str(out), repr(e))


def read_jobs(path):
    jobs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            n_pairs, seed, outp = parts[0], parts[1], parts[2]
            jobs.append((int(n_pairs), int(seed), outp))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--jobs_file", default=JOBS_FILE)
    ap.add_argument("--state", default=STATE_FILE)
    args = ap.parse_args()

    if not Path(args.jobs_file).exists():
        print(f"ERROR: {args.jobs_file} not found. "
              f"Run `python exp0_fete_baseline.py` first.")
        sys.exit(1)
    if not Path(args.state).exists():
        print(f"ERROR: {args.state} not found.")
        sys.exit(1)

    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

    all_jobs = read_jobs(args.jobs_file)
    todo = [j for j in all_jobs if not Path(j[2]).exists()]
    print(f"Workers: {n_workers}   "
          f"Jobs: {len(all_jobs)} total, {len(all_jobs)-len(todo)} done, "
          f"{len(todo)} to run")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0
    with mp.Pool(processes=n_workers, initializer=_init_worker,
                 initargs=(args.state,)) as pool:
        try:
            for res in pool.imap_unordered(_do_job, todo, chunksize=1):
                completed += 1
                if res[0] == "done":
                    _, out, auc, m2, npairs = res
                    print(f"[{completed:3d}/{len(todo)}]  n_pairs={npairs:4d}  "
                          f"AUC={auc:.3f}  m2={m2:.3f}", flush=True)
                elif res[0] == "error":
                    print(f"[{completed:3d}/{len(todo)}]  ERROR: {res[2]}",
                          flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            pool.terminate()
            pool.join()
            sys.exit(130)

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s. Next: python analyze_fete.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()
