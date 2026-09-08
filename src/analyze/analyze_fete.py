#!/usr/bin/env python3
"""
analyze_fete.py
Aggregates FETE baseline runs and produces the comparison table
requested by JCAA reviewer 2.

Reports:
    - FETE performance (AUC, m2, LCC) as a function of n_pairs, with
      variance across seeds
    - Comparison against the CA canonical configuration
      (eps=0.15, kappa=0.03) if that CA point exists in results_phase1/

Outputs:
    fete_summary.csv
    fete_vs_ca.png            side-by-side comparison
    fete_summary.txt          text summary for the reviewer response

Usage:
    python analyze_fete.py [--fete_dir DIR] [--ca_dir DIR]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9


def load_dir(d):
    rows = []
    for f in sorted(Path(d).glob("*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    # Kernel emits 'n_random_pairs'; alias to 'n_pairs' for readability
    if 'n_random_pairs' in df.columns and 'n_pairs' not in df.columns:
        df = df.rename(columns={'n_random_pairs': 'n_pairs'})
    return df


def summarise_fete(df):
    """Mean and std per n_pairs."""
    g = df.groupby('n_pairs')
    return g.agg(
        auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
        m1_mean=('m1', 'mean'),   m1_std=('m1', 'std'),
        m2_mean=('m2', 'mean'),   m2_std=('m2', 'std'),
        m3_mean=('m3', 'mean'),   m3_std=('m3', 'std'),
        lcc_mean=('lcc_size', 'mean'), lcc_std=('lcc_size', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()


def summarise_ca_canonical(df):
    """Extract stats for eps=0.15, kappa=0.03 point specifically."""
    canon = df[np.isclose(df['eps'], 0.15) & np.isclose(df['kappa'], 0.03)]
    if len(canon) == 0:
        return None
    return {
        'auc_mean': canon['auc'].mean(), 'auc_std': canon['auc'].std(),
        'm1_mean': canon['m1'].mean(),   'm1_std': canon['m1'].std(),
        'm2_mean': canon['m2'].mean(),   'm2_std': canon['m2'].std(),
        'm3_mean': canon['m3'].mean(),   'm3_std': canon['m3'].std(),
        'lcc_mean': canon['lcc_size'].mean(),
        'lcc_std': canon['lcc_size'].std(),
        'n_seeds': len(canon),
    }


def comparison_plot(fete_summary, ca_stats, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    npairs = fete_summary['n_pairs'].values

    # Panel 1: AUC
    ax = axes[0]
    ax.errorbar(npairs, fete_summary['auc_mean'],
                yerr=fete_summary['auc_std'], fmt='o-',
                color='steelblue', capsize=3, label='FETE baseline')
    if ca_stats:
        ax.axhline(ca_stats['auc_mean'], color='crimson',
                   ls='--', lw=1.5, label='CA canonical (ε=0.15, κ=0.03)')
        ax.fill_between([npairs.min(), npairs.max()],
                        ca_stats['auc_mean'] - ca_stats['auc_std'],
                        ca_stats['auc_mean'] + ca_stats['auc_std'],
                        color='crimson', alpha=0.15)
    ax.set_xlabel('n random pairs (FETE)')
    ax.set_ylabel('AUC')
    ax.set_xscale('log')
    ax.set_title('Ranking performance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: m2 (traffic condensation)
    ax = axes[1]
    ax.errorbar(npairs, fete_summary['m2_mean'],
                yerr=fete_summary['m2_std'], fmt='o-',
                color='steelblue', capsize=3, label='FETE')
    if ca_stats:
        ax.axhline(ca_stats['m2_mean'], color='crimson',
                   ls='--', lw=1.5, label='CA canonical')
        ax.fill_between([npairs.min(), npairs.max()],
                        ca_stats['m2_mean'] - ca_stats['m2_std'],
                        ca_stats['m2_mean'] + ca_stats['m2_std'],
                        color='crimson', alpha=0.15)
    ax.set_xlabel('n random pairs (FETE)')
    ax.set_ylabel(r'$m_2$ (fraction of traffic in LCC)')
    ax.set_xscale('log')
    ax.set_title('Corridor condensation')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: LCC size
    ax = axes[2]
    ax.errorbar(npairs, fete_summary['lcc_mean'],
                yerr=fete_summary['lcc_std'], fmt='o-',
                color='steelblue', capsize=3, label='FETE')
    if ca_stats:
        ax.axhline(ca_stats['lcc_mean'], color='crimson',
                   ls='--', lw=1.5, label='CA canonical')
        ax.fill_between([npairs.min(), npairs.max()],
                        ca_stats['lcc_mean'] - ca_stats['lcc_std'],
                        ca_stats['lcc_mean'] + ca_stats['lcc_std'],
                        color='crimson', alpha=0.15)
    ax.set_xlabel('n random pairs (FETE)')
    ax.set_ylabel(r'$|C_1|$ (LCC size, cells)')
    ax.set_xscale('log')
    ax.set_title('Backbone size')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fete_dir", default="results_fete")
    ap.add_argument("--ca_dir", default="results_phase1")
    args = ap.parse_args()

    df_fete = load_dir(args.fete_dir)
    df_ca = load_dir(args.ca_dir)

    print(f"Loaded {len(df_fete)} FETE runs, {len(df_ca)} CA runs")
    if len(df_fete) == 0:
        print("No FETE results yet.")
        return

    fete_sum = summarise_fete(df_fete)
    fete_sum.to_csv("fete_summary.csv", index=False)
    print("Wrote fete_summary.csv")

    ca_stats = None
    if len(df_ca) > 0:
        ca_stats = summarise_ca_canonical(df_ca)

    comparison_plot(fete_sum, ca_stats, "fete_vs_ca.png")

    lines = ["# FETE vs CA comparison  (response to JCAA reviewer 2)"]
    lines.append("")
    lines.append("## FETE baseline")
    lines.append(fete_sum.to_string(index=False, float_format='{:.4f}'.format))

    if ca_stats:
        lines.append("")
        lines.append("## CA canonical (ε=0.15, κ=0.03) for reference")
        lines.append(f"  AUC  = {ca_stats['auc_mean']:.4f} ± {ca_stats['auc_std']:.4f}")
        lines.append(f"  m2   = {ca_stats['m2_mean']:.4f} ± {ca_stats['m2_std']:.4f}")
        lines.append(f"  LCC  = {ca_stats['lcc_mean']:.1f} ± {ca_stats['lcc_std']:.1f}")
        lines.append(f"  based on {ca_stats['n_seeds']} seeds")

        # Direct comparison at matched n_pairs = 86
        m86 = fete_sum[fete_sum['n_pairs'] == 86]
        if len(m86):
            row = m86.iloc[0]
            lines.append("")
            lines.append("## Head-to-head at n_pairs = 86 (matched budget)")
            lines.append(f"  FETE : AUC = {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
            lines.append(f"  CA   : AUC = {ca_stats['auc_mean']:.4f} ± {ca_stats['auc_std']:.4f}")
            delta = ca_stats['auc_mean'] - row['auc_mean']
            lines.append(f"  ΔAUC (CA - FETE) = {delta:+.4f}")
            lines.append("")
            lines.append(f"  FETE : m2  = {row['m2_mean']:.4f} ± {row['m2_std']:.4f}")
            lines.append(f"  CA   : m2  = {ca_stats['m2_mean']:.4f} ± {ca_stats['m2_std']:.4f}")

    with open("fete_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("\nWrote fete_summary.txt")


if __name__ == "__main__":
    main()
