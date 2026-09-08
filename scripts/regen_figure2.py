#!/usr/bin/env python3
"""
regen_figure2.py
Regenerates Figure 2 (6-panel layers/surfaces) from the SAME grid CSV
used by CA4_7.py (grilla_con_W_hidrica.csv), NOT from the raw DEM.

This addresses reviewer 1's concern that Fig. 2 and Fig. 3 in the
submitted paper had different extents. Both figures now use the exact
same 12,826 valid cells, guaranteeing consistency.

Usage:
    python regen_figure2.py --csv grilla_con_W_hidrica.csv \\
        --out figure2_corrected.png
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figure2_corrected.png")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from ca_kernel import build_state

    # Use the same state pipeline as the CA model. This guarantees the
    # extent, grid geometry, and normalisation are identical to Fig. 3.
    print("Building state from CSV (same pipeline as CA4_7.py)...")
    state = build_state(args.csv)
    print(f"  valid cells: {int(state.valid_mask.sum())}")
    print(f"  sites      : {state.n_sites}")

    # Grids
    h_grid = state.h_grid
    p_grid = state.p_grid
    w_grid = state.w_grid
    T_grid = state.T_grid
    E_grid = state.E_grid
    c_grid = state.c_grid
    valid_mask = state.valid_mask

    # Mask out invalid cells (make them NaN so they render as blank)
    def masked(arr):
        out = np.where(valid_mask, arr, np.nan)
        return out

    panels = [
        (masked(h_grid), r'$\tilde{h}$ (normalised elevation)'),
        (masked(p_grid), r'$\tilde{p}$ (normalised slope)'),
        (masked(w_grid), r'$W$ (hydrological corridor)'),
        (masked(T_grid), r'$T$ (terrace suitability)'),
        (masked(E_grid), r'$E$ (environmental affinity)'),
        (masked(c_grid), r'$c$ (cost surface)'),
    ]

    # Tight bounding box of valid extent for cropping
    rows_v = np.where(valid_mask.any(axis=1))[0]
    cols_v = np.where(valid_mask.any(axis=0))[0]
    r0, r1 = rows_v[0], rows_v[-1] + 1
    c0, c1 = cols_v[0], cols_v[-1] + 1

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0))
    last_im = None
    for ax, (arr, title) in zip(axes.flat, panels):
        crop = arr[r0:r1, c0:c1]
        # imshow with origin='upper' matches array convention;
        # y-axis reversed so North is up when Y = northing (UTM)
        im = ax.imshow(crop, cmap='viridis', origin='lower',
                        interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('0.4')
        last_im = im

    # Add a small scalebar to the bottom-left panel
    # 250 m per cell → find the number of cells for 5 km
    scale_km = 5
    cells_5km = int(scale_km * 1000 / 250)  # = 20 cells
    ax0 = axes[1, 0]
    ny_c = r1 - r0
    nx_c = c1 - c0
    x_start = int(0.05 * nx_c)
    x_end = x_start + cells_5km
    y_pos = int(0.92 * ny_c)
    ax0.plot([x_start, x_end], [y_pos, y_pos],
              color='white', lw=3, solid_capstyle='butt')
    ax0.text((x_start + x_end) / 2, y_pos - int(0.03 * ny_c),
              f'{scale_km} km', color='white', ha='center',
              va='bottom', fontsize=9, fontweight='bold')

    # North arrow (top-left panel)
    ax_n = axes[0, 0]
    ax_n.annotate('N', xy=(int(0.95 * nx_c), int(0.10 * ny_c)),
                   xytext=(int(0.95 * nx_c), int(0.25 * ny_c)),
                   arrowprops=dict(arrowstyle='-|>', color='white', lw=2),
                   color='white', fontsize=11, fontweight='bold',
                   ha='center')

    plt.subplots_adjust(left=0.02, right=0.92, top=0.96, bottom=0.02,
                         hspace=0.12, wspace=0.06)

    cbar_ax = fig.add_axes([0.935, 0.18, 0.012, 0.65])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label('Normalised value', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight',
                 facecolor='white')
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
