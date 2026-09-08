#!/usr/bin/env python3
"""
analyze_phase1.py
Aggregates the JSON outputs of Phase 1 and produces the diagnostic
plots needed for the go/no-go decision before launching Phase 2.

Outputs:
    phase1_aggregate.csv         per-(eps,kappa) means and variances
    phase1_overview.png           4 panels: <m1>, <m2>, <m3>, <AUC>
    phase1_susceptibilities.png   chi_1, chi_2, chi_3
    phase1_ridge_detection.png    profile cuts; candidate critical line
    phase1_summary.txt            text summary with go/no-go verdict

Usage:
    python analyze_phase1.py [--results_dir DIR]
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


def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows)


def aggregate(df):
    g = df.groupby(['eps', 'kappa'])
    agg = g.agg(
        m1_mean=('m1', 'mean'), m1_var=('m1', 'var'),
        m2_mean=('m2', 'mean'), m2_var=('m2', 'var'),
        m3_mean=('m3', 'mean'), m3_var=('m3', 'var'),
        auc_mean=('auc', 'mean'), auc_var=('auc', 'var'),
        lcc_mean=('lcc_size', 'mean'), lcc_var=('lcc_size', 'var'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    return agg


def pivot_to_grid(agg, col):
    return agg.pivot(index='kappa', columns='eps', values=col)


def overview_plot(agg, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ('m1_mean',  r'$\langle m_1 \rangle$  (cell fraction in LCC)',  'viridis'),
        ('m2_mean',  r'$\langle m_2 \rangle$  (traffic fraction in LCC)','viridis'),
        ('m3_mean',  r'$\langle m_3 \rangle = e^{-S}$  (concentration)','viridis'),
        ('auc_mean', r'$\langle \mathrm{AUC} \rangle$',                 'plasma'),
    ]
    for ax, (col, title, cmap) in zip(axes.flat, panels):
        Z = pivot_to_grid(agg, col)
        im = ax.imshow(
            Z.values, origin='lower', aspect='auto',
            extent=[Z.columns.min(), Z.columns.max(),
                    Z.index.min(), Z.index.max()],
            cmap=cmap,
        )
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel(r'$\kappa$')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def susceptibility_plot(agg, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ('m1_var', r'$\chi_1 = \mathrm{Var}(m_1)$'),
        ('m2_var', r'$\chi_2 = \mathrm{Var}(m_2)$'),
        ('m3_var', r'$\chi_3 = \mathrm{Var}(m_3)$'),
    ]
    for ax, (col, title) in zip(axes, panels):
        Z = pivot_to_grid(agg, col)
        Zlog = np.log10(Z.values + 1e-12)
        im = ax.imshow(
            Zlog, origin='lower', aspect='auto',
            extent=[Z.columns.min(), Z.columns.max(),
                    Z.index.min(), Z.index.max()],
            cmap='inferno',
        )
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel(r'$\kappa$')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r'$\log_{10}(\chi)$')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def ridge_detection(agg, out_path, summary_lines):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    kappas = sorted(agg['kappa'].unique())
    n_show = min(6, len(kappas))
    show_idx = np.linspace(0, len(kappas) - 1, n_show).astype(int)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_show))
    ax = axes[0]
    for c, ki in zip(cmap, show_idx):
        kap = kappas[ki]
        sub = agg[np.isclose(agg['kappa'], kap)].sort_values('eps')
        ax.plot(sub['eps'], sub['m2_var'], 'o-', color=c,
                label=fr'$\kappa={kap:.3f}$')
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\chi_2 = \mathrm{Var}(m_2)$')
    ax.set_title(r'Susceptibility profiles')
    ax.legend(fontsize=7, loc='best')
    ax.set_yscale('log')

    ridge = []
    for kap in kappas:
        sub = agg[np.isclose(agg['kappa'], kap)].sort_values('eps')
        if sub['m2_var'].max() <= 0:
            continue
        i_max = sub['m2_var'].values.argmax()
        ridge.append((kap, sub['eps'].iloc[i_max],
                       sub['m2_var'].iloc[i_max]))
    ridge = np.array(ridge)

    ax = axes[1]
    if len(ridge):
        ax.plot(ridge[:, 1], ridge[:, 0], 'o-', color='C3',
                markersize=8, lw=2, label=r'ridge of $\chi_2$')
        ax.set_xlim(agg['eps'].min(), agg['eps'].max())
        ax.set_ylim(agg['kappa'].min(), agg['kappa'].max())
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\kappa$')
    ax.set_title(r'Candidate critical ridge')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    summary_lines.append("\n--- Ridge detection ---")
    if len(ridge):
        summary_lines.append(
            f"  Found {len(ridge)} ridge points in (eps, kappa) plane")
        summary_lines.append(
            f"  eps_c range: {ridge[:,1].min():.3f} to {ridge[:,1].max():.3f}")
        eps_c = ridge[:, 1]
        n_interior = np.sum(
            (eps_c > agg['eps'].min() + 1e-6) &
            (eps_c < agg['eps'].max() - 1e-6))
        summary_lines.append(
            f"  Interior ridge points: {n_interior} / {len(ridge)}")
        if n_interior >= 5:
            summary_lines.append(
                "  ✓ Ridge appears WELL-DEFINED (likely transition)")
        elif n_interior >= 2:
            summary_lines.append(
                "  ? Ridge AMBIGUOUS (may be transition or crossover)")
        else:
            summary_lines.append(
                "  ✗ Ridge PINNED to grid edge (no transition in scanned range)")
    else:
        summary_lines.append("  No ridge could be located.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_phase1")
    args = ap.parse_args()

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} runs from {args.results_dir}")
    if len(df) == 0:
        print("No results yet. Run Phase 1 first.")
        return
    expected = 11 * 11 * 10
    if len(df) < expected:
        print(f"  WARNING: expected {expected}, missing "
              f"{expected - len(df)} runs")

    agg = aggregate(df)
    agg.to_csv("phase1_aggregate.csv", index=False)
    print(f"Wrote phase1_aggregate.csv  ({len(agg)} grid points)")

    summary = ["# Phase 1 Summary"]
    summary.append(f"Total runs : {len(df)}")
    summary.append(f"Grid points: {len(agg)}")
    summary.append("")
    summary.append("--- Order parameter ranges ---")
    for m in ('m1', 'm2', 'm3'):
        col = f'{m}_mean'
        summary.append(f"  {m}: [{agg[col].min():.4f}, {agg[col].max():.4f}]"
                       f"  (range = {agg[col].max() - agg[col].min():.4f})")
    summary.append(f"  AUC: [{agg['auc_mean'].min():.4f}, "
                   f"{agg['auc_mean'].max():.4f}]")

    overview_plot(agg, "phase1_overview.png")
    susceptibility_plot(agg, "phase1_susceptibilities.png")
    ridge_detection(agg, "phase1_ridge_detection.png", summary)

    txt = "\n".join(summary)
    with open("phase1_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("\nWrote phase1_summary.txt")


if __name__ == "__main__":
    main()
