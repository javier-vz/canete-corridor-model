#!/usr/bin/env python3
"""
build_state.py
Builds the static state ONCE (lattice, base graph, attractors, pairs)
and pickles it. Run this once per cluster environment before launching
phase scans or FETE baseline.

Usage:
    python build_state.py --csv grilla_con_W_hidrica.csv --out state_cache.pkl
"""
import argparse
import pickle
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="state_cache.pkl")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from ca_kernel import build_state

    print(f"Building state from {args.csv}...")
    t0 = time.time()
    state = build_state(args.csv)
    dt = time.time() - t0

    print(f"Built in {dt:.1f}s:")
    print(f"  sites         : {state.n_sites}")
    print(f"  attractors    : {len(state.attractors)}")
    print(f"  candidate pairs: {len(state.pair_candidates)}")
    print(f"  graph nodes   : {state.G_template.number_of_nodes()}")
    print(f"  graph edges   : {state.G_template.number_of_edges()}")

    with open(args.out, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
