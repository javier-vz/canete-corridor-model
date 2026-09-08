#!/usr/bin/env python3
"""
run_single.py
Runs ONE CA point OR one FETE baseline and writes result to JSON.

Usage - CA:
    python run_single.py ca --eps 0.15 --kappa 0.03 --seed 42 \\
        --state state_cache.pkl --out results/run_e0.15_k0.03_s42.json

Usage - FETE baseline:
    python run_single.py fete --n_pairs 86 --seed 42 \\
        --state state_cache.pkl --out results/fete_n86_s42.json

Designed for cluster array jobs. Each job is independent and idempotent
(skips if output file already exists).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _run_ca(state, args):
    from ca_kernel import run_one
    return run_one(state,
                    eps=args.eps, kappa=args.kappa, seed=args.seed,
                    beta_path=args.beta,
                    n_paths=args.n_paths,
                    topK_mult=args.topK_mult,
                    congestion_cap=args.congestion_cap)


def _run_fete(state, args):
    from ca_kernel import run_fete
    return run_fete(state,
                     n_random_pairs=args.n_pairs,
                     seed=args.seed,
                     topK_mult=args.topK_mult)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # CA subparser
    ca = sub.add_parser("ca", help="Run CA model")
    ca.add_argument("--eps", type=float, required=True)
    ca.add_argument("--kappa", type=float, required=True)
    ca.add_argument("--seed", type=int, required=True)
    ca.add_argument("--beta", type=float, default=1.5)
    ca.add_argument("--n_paths", type=int, default=4)
    ca.add_argument("--congestion_cap", type=float, default=2.0)
    ca.add_argument("--topK_mult", type=float, default=2.0)

    # FETE subparser
    fete = sub.add_parser("fete", help="Run FETE baseline")
    fete.add_argument("--n_pairs", type=int, required=True)
    fete.add_argument("--seed", type=int, required=True)
    fete.add_argument("--topK_mult", type=float, default=2.0)

    ap.add_argument("--state", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[skip] {out} exists", flush=True)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    with open(args.state, "rb") as f:
        state = pickle.load(f)

    t0 = time.time()
    if args.mode == "ca":
        res = _run_ca(state, args)
    elif args.mode == "fete":
        res = _run_fete(state, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    dt = time.time() - t0
    res["runtime_s"] = dt
    res["host"] = os.uname().nodename

    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    if args.mode == "ca":
        print(f"[done] {out}  ({dt:.1f}s)  m1={res['m1']:.3f} "
              f"m2={res['m2']:.3f} AUC={res['auc']:.3f}", flush=True)
    else:
        print(f"[done fete] {out}  ({dt:.1f}s)  m2={res['m2']:.3f} "
              f"AUC={res['auc']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
