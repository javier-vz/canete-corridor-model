#!/usr/bin/env python3
"""
exp1_phase_scan.py
Generates the coarse (eps, kappa, seed) grid for Phase 1.

Output:
    phase1_jobs.txt      one line per job: "eps kappa seed out_path"
    submit_phase1.sh     SLURM array submission
    run_phase1_parallel.sh   GNU parallel alternative

Grid: 11x11 x 10 seeds = 1,210 runs (~25 min at 100 cores).
"""
import numpy as np
from pathlib import Path

EPS_VALUES = np.linspace(0.0, 0.40, 11)
KAP_VALUES = np.linspace(0.0, 0.10, 11)
N_SEEDS    = 10
SEED_BASE  = 1000

OUT_DIR = Path("results_phase1")
JOBS_FILE = "phase1_jobs.txt"
SUBMIT_FILE = "submit_phase1.sh"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    jobs = []
    for eps in EPS_VALUES:
        for kap in KAP_VALUES:
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                fname = f"e{eps:.3f}_k{kap:.3f}_s{seed}.json"
                out = OUT_DIR / fname
                jobs.append((float(eps), float(kap), seed, str(out)))

    with open(JOBS_FILE, "w") as f:
        for eps, kap, seed, out in jobs:
            f.write(f"{eps:.4f} {kap:.4f} {seed} {out}\n")

    print(f"Wrote {len(jobs)} jobs to {JOBS_FILE}")
    print(f"  eps  : {len(EPS_VALUES)} values "
          f"({EPS_VALUES[0]} to {EPS_VALUES[-1]})")
    print(f"  kappa: {len(KAP_VALUES)} values "
          f"({KAP_VALUES[0]} to {KAP_VALUES[-1]})")
    print(f"  seeds: {N_SEEDS} per (eps,kappa) point")
    print(f"  total: {len(jobs)} runs")
    print(f"  est wall @ 100 cores: ~{len(jobs)*120/100/60:.0f} min")
    print(f"  est wall @ 200 cores: ~{len(jobs)*120/200/60:.0f} min")

    submit = f"""#!/bin/bash
#SBATCH --job-name=ca_phase1
#SBATCH --output=slurm_logs/phase1_%A_%a.out
#SBATCH --error=slurm_logs/phase1_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{len(jobs)}%200

mkdir -p slurm_logs

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {JOBS_FILE})
read -r EPS KAP SEED OUT <<< "$LINE"

python run_single.py ca \\
    --eps "$EPS" --kappa "$KAP" --seed "$SEED" \\
    --state state_cache.pkl --out "$OUT"
"""
    with open(SUBMIT_FILE, "w") as f:
        f.write(submit)
    Path(SUBMIT_FILE).chmod(0o755)
    print(f"\nWrote {SUBMIT_FILE} (SLURM)")

    parallel = f"""#!/bin/bash
# GNU parallel alternative (no scheduler).
# Usage: ./run_phase1_parallel.sh [NCORES]   default = $(nproc)
NCORES=${{1:-$(nproc)}}
mkdir -p logs
echo "Running {len(jobs)} jobs on $NCORES cores"
awk '{{print $1, $2, $3, $4}}' {JOBS_FILE} | \\
    parallel --colsep ' ' --jobs "$NCORES" \\
    --joblog logs/parallel_phase1.log \\
    python run_single.py ca --eps {{1}} --kappa {{2}} --seed {{3}} \\
        --state state_cache.pkl --out {{4}}
"""
    with open("run_phase1_parallel.sh", "w") as f:
        f.write(parallel)
    Path("run_phase1_parallel.sh").chmod(0o755)
    print(f"Wrote run_phase1_parallel.sh (GNU parallel)")


if __name__ == "__main__":
    main()
