#!/usr/bin/env python3
"""
preview_phase1.py

Non-invasive preview of Phase 1 while the runner is still going.
Reads whatever JSONs exist in results_phase1/ and shows a partial
picture. Safe to run any number of times — does NOT interfere with
run_phase1_local.py.

Usage:
    python preview_phase1.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results_phase1")


def main():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No results yet in results_phase1/")
        return

    rows = []
    for f in files:
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception:
            pass  # Skip files that are being written

    df = pd.DataFrame(rows)
    total_expected = 11 * 11 * 10  # 1,210
    pct = 100 * len(df) / total_expected
    print(f"\n{len(df)}/{total_expected} runs completed ({pct:.1f}%)")

    if len(df) < 50:
        print("Too few points for a meaningful preview.")
        return

    print("\n--- Order parameter ranges so far ---")
    for m in ('m1', 'm2', 'm3', 'auc'):
        if m in df.columns:
            print(f"  {m}: [{df[m].min():.4f}, {df[m].max():.4f}]  "
                  f"mean={df[m].mean():.4f}")

    print("\n--- Coverage of the (eps, kappa) grid ---")
    g = df.groupby(['eps', 'kappa']).size().reset_index(name='n_seeds')
    print(f"  Grid points touched: {len(g)} / 121")
    fully_done = g[g['n_seeds'] == 10]
    print(f"  Points with all 10 seeds done: {len(fully_done)} / 121")

    # Where in the grid are we?
    eps_done = sorted(df['eps'].unique())
    kap_done = sorted(df['kappa'].unique())
    print(f"\n  eps values seen: {[f'{v:.2f}' for v in eps_done]}")
    print(f"  kap values seen: {[f'{v:.3f}' for v in kap_done]}")

    # Rough phase-diagram-in-progress (only points with >= 3 seeds)
    ok_points = g[g['n_seeds'] >= 3]
    if len(ok_points) >= 10:
        print(f"\n--- Rough phase diagram (points with >= 3 seeds only) ---")
        agg = (df.groupby(['eps', 'kappa'])
                 .agg(m2_mean=('m2', 'mean'), n=('seed', 'count'))
                 .reset_index())
        agg = agg[agg['n'] >= 3]
        # print a text heatmap
        pivot = agg.pivot(index='kappa', columns='eps', values='m2_mean')
        print("\n  <m2> over (eps rows, kap cols) -- '.' = insufficient data")
        print("  eps :", end="")
        for eps in sorted(df['eps'].unique()):
            print(f"  {eps:.2f}", end="")
        print()
        for kap in sorted(df['kappa'].unique()):
            print(f"  {kap:.3f}:", end="")
            for eps in sorted(df['eps'].unique()):
                try:
                    v = pivot.loc[kap, eps]
                    if pd.isna(v):
                        print(f"    . ", end="")
                    else:
                        print(f"  {v:.2f}", end="")
                except KeyError:
                    print(f"    . ", end="")
            print()


if __name__ == "__main__":
    main()
