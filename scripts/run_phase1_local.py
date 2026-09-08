#!/usr/bin/env python3
"""
run_phase1_local.py

Local Windows/Anaconda runner for the Phase 1 phase scan.
Uses Python multiprocessing (no bash, no SLURM, no GNU parallel).

Runs 1,210 jobs (11 eps x 11 kappa x 10 seeds) using all available
cores minus one (to keep the machine responsive).

Idempotent: skips jobs whose output JSON already exists. Safe to
interrupt with Ctrl+C and resume by re-running.

Usage:
    python run_phase1_local.py                # auto-detect cores
    python run_phase1_local.py --workers 6    # use 6 workers
    python run_phase1_local.py --dry_run      # just count jobs
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
from pathlib import Path

# --------------------------------------------------------------
# Windows note: multiprocessing on Windows uses "spawn" start
# method by default, so the child process re-imports this script.
# We MUST put the top-level runnable code inside `if __name__ == '__main__'`
# and keep the worker function importable at module scope.
# --------------------------------------------------------------

JOBS_FILE = "phase1_jobs.txt"
STATE_FILE = "state_cache.pkl"


def _init_worker(state_path):
    """Load state ONCE per worker process to avoid reloading 5 MB
    of pickle for every job."""
    global _WORKER_STATE
    global _WORKER_KERNEL
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ca_kernel import run_one as _run_one
    _WORKER_KERNEL = _run_one
    with open(state_path, "rb") as f:
        _WORKER_STATE = pickle.load(f)


def _do_job(job):
    """Runs one (eps, kappa, seed) point. Returns None on success or
    error string on failure."""
    eps, kappa, seed, out_path = job
    out = Path(out_path)
    if out.exists():
        return ("skip", str(out))
    try:
        t0 = time.time()
        res = _WORKER_KERNEL(_WORKER_STATE,
                              eps=eps, kappa=kappa, seed=seed)
        res["runtime_s"] = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        return ("done", str(out), res["auc"], res["m1"], res["m2"])
    except Exception as e:
        return ("error", str(out), repr(e))


def read_jobs(path):
    jobs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            eps, kap, seed, outp = parts[0], parts[1], parts[2], parts[3]
            jobs.append((float(eps), float(kap), int(seed), outp))
    return jobs


def _format_hms(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers. Default: cpu_count-1")
    ap.add_argument("--jobs_file", default=JOBS_FILE)
    ap.add_argument("--state", default=STATE_FILE)
    ap.add_argument("--dry_run", action="store_true",
                    help="Just report how many jobs remain")
    args = ap.parse_args()

    if not Path(args.jobs_file).exists():
        print(f"ERROR: {args.jobs_file} not found. "
              f"Did you run `python exp1_phase_scan.py` first?")
        sys.exit(1)
    if not Path(args.state).exists():
        print(f"ERROR: {args.state} not found. "
              f"Did you run `python build_state.py --csv ...`?")
        sys.exit(1)

    n_cores = mp.cpu_count()
    n_workers = args.workers if args.workers > 0 else max(1, n_cores - 1)

    all_jobs = read_jobs(args.jobs_file)
    todo = [j for j in all_jobs if not Path(j[3]).exists()]
    done_already = len(all_jobs) - len(todo)

    print(f"System : {n_cores} logical cores available")
    print(f"Workers: {n_workers}")
    print(f"Jobs   : {len(all_jobs)} total, {done_already} already done, "
          f"{len(todo)} to run")

    if len(todo) == 0:
        print("\nNothing to do. All jobs are already complete.")
        return

    if args.dry_run:
        eta_serial = len(todo) * 120
        eta_par = eta_serial / n_workers
        print(f"\nEstimated wall time: {_format_hms(eta_par)} "
              f"(~{_format_hms(eta_serial)} serial)")
        return

    t_start = time.time()
    completed = 0
    errors = 0

    # Windows-safe pool with worker initializer
    with mp.Pool(processes=n_workers,
                 initializer=_init_worker,
                 initargs=(args.state,)) as pool:
        try:
            for res in pool.imap_unordered(_do_job, todo, chunksize=1):
                completed += 1
                if res[0] == "done":
                    _, out, auc, m1, m2 = res
                    tag = "OK"
                    detail = f"AUC={auc:.3f} m1={m1:.3f} m2={m2:.3f}"
                elif res[0] == "skip":
                    tag = "SKIP"
                    detail = ""
                else:
                    errors += 1
                    tag = "ERR"
                    detail = res[2]
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 1e-6)
                remaining = (len(todo) - completed) / max(rate, 1e-6)
                pct = 100.0 * completed / len(todo)
                print(f"[{completed:4d}/{len(todo)}] "
                      f"{pct:5.1f}%  {tag:4s}  "
                      f"elapsed {_format_hms(elapsed)}  "
                      f"eta {_format_hms(remaining)}  "
                      f"{detail}", flush=True)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved (idempotent).")
            print(f"Completed this session: {completed}/{len(todo)}")
            pool.terminate()
            pool.join()
            sys.exit(130)

    total_elapsed = time.time() - t_start
    print(f"\nDone. {completed} completed in {_format_hms(total_elapsed)}")
    if errors:
        print(f"  {errors} errors (check output above)")
    print(f"\nNext: python analyze_phase1.py")


if __name__ == "__main__":
    # This guard is CRITICAL on Windows for multiprocessing.
    mp.freeze_support()
    main()
