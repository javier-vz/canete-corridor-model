#!/usr/bin/env python3
"""
exp0_fete_baseline.py
Generates the job list for the FETE baseline experiment.

This is the empirical comparison requested by JCAA reviewer 2:
"How is this approach comparable to for example, using
From-Everywhere-to-Everywhere?"

Design: for each n_pairs in {50, 86, 200, 500, 1000}, run 30 seeds
using random endpoint sampling with deterministic LCP (no eps, no kappa).
Total: 150 runs (~5-10 min at 100 cores because FETE is faster than CA).

Output:
    fete_jobs.txt          job manifest
    submit_fete.sh         SLURM
    run_fete_parallel.sh   GNU parallel

n_pairs = 86 matches the number of local attractor pairs in the CA
model, so the direct comparison is fair in terms of computational
budget.
"""
import numpy as np
from pathlib import Path

N_PAIRS_VALUES = [50, 86, 200, 500, 1000]
N_SEEDS        = 30
SEED_BASE      = 2000

OUT_DIR = Path("results_fete")
JOBS_FILE = "fete_jobs.txt"
SUBMIT_FILE = "submit_fete.sh"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    jobs = []
    for n_pairs in N_PAIRS_VALUES:
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            fname = f"fete_n{n_pairs}_s{seed}.json"
            out = OUT_DIR / fname
            jobs.append((n_pairs, seed, str(out)))

    with open(JOBS_FILE, "w") as f:
        for n_pairs, seed, out in jobs:
            f.write(f"{n_pairs} {seed} {out}\n")

    print(f"Wrote {len(jobs)} jobs to {JOBS_FILE}")
    print(f"  n_pairs values: {N_PAIRS_VALUES}")
    print(f"  seeds per point: {N_SEEDS}")
    print(f"  total: {len(jobs)} runs")
    # FETE is roughly 10-40x faster than CA depending on n_pairs
    # (single LCP per pair, no k-shortest ensemble, no perturbations)
    est_avg = np.mean([n * 0.15 for n in N_PAIRS_VALUES]) * N_SEEDS
    print(f"  est wall @ 100 cores: ~{est_avg/60:.0f}-{est_avg*2/60:.0f} min")

    submit = f"""#!/bin/bash
#SBATCH --job-name=fete_baseline
#SBATCH --output=slurm_logs/fete_%A_%a.out
#SBATCH --error=slurm_logs/fete_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{len(jobs)}%150

mkdir -p slurm_logs

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {JOBS_FILE})
read -r N_PAIRS SEED OUT <<< "$LINE"

python run_single.py fete \\
    --n_pairs "$N_PAIRS" --seed "$SEED" \\
    --state state_cache.pkl --out "$OUT"
"""
    with open(SUBMIT_FILE, "w") as f:
        f.write(submit)
    Path(SUBMIT_FILE).chmod(0o755)
    print(f"\nWrote {SUBMIT_FILE} (SLURM)")

    parallel = f"""#!/bin/bash
# GNU parallel alternative.
NCORES=${{1:-$(nproc)}}
mkdir -p logs
echo "Running {len(jobs)} jobs on $NCORES cores"
awk '{{print $1, $2, $3}}' {JOBS_FILE} | \\
    parallel --colsep ' ' --jobs "$NCORES" \\
    --joblog logs/parallel_fete.log \\
    python run_single.py fete --n_pairs {{1}} --seed {{2}} \\
        --state state_cache.pkl --out {{3}}
"""
    with open("run_fete_parallel.sh", "w") as f:
        f.write(parallel)
    Path("run_fete_parallel.sh").chmod(0o755)
    print(f"Wrote run_fete_parallel.sh (GNU parallel)")


if __name__ == "__main__":
    main()
