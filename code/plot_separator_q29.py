"""
plot_separator_q29.py
Generate the fibre separator figure for the Q1 paper.

Computes residual_rank = rank((I - P_qc) V_c), the rank of the orthogonal
projection of V_n^(c) onto (V_n^(q-c))^perp -- NOT the set intersection
V_n^(c) cap (V_n^(q-c))^perp (see quantum-structure/notes/q1-central-theorem-
audit.md, 2026-08-05: these are different objects, confirmed by an explicit
countermodel). residual_rank == 0 still rigorously proves V_c == V_qc exactly
(by rank-nullity: dim V_c = dim(V_c cap V_qc) + residual_rank, combined with
dim V_c == dim V_qc always). Computed at each BFS depth for q=29 and four
conjugate pairs, alongside the per-pair central-phase coherence ell_gamma^(c)(n)
and fractional saturation rank V_n^(c)/q.

Requires:
    spectral_O12.py  (Weil block pipeline)
    separator.py     (GS separator functional)

Usage:
    python plot_separator_q29.py
    python plot_separator_q29.py --q 29 --bfs-frac 0.50 --output separator_q29

Outputs:
    separator_q29_transition.pdf
    separator_q29_transition.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectral_O12 import (
    build_generators,
    bfs_shells,
    fingerprint_vectors_batch,
    coherence_length,
    EPS_GS,
)
from separator import separator_scores


def compute_separator_profile(q: int, pairs: list[tuple[int, int]],
                               bfs_frac: float) -> dict:
    """
    Compute residual_rank, ell_gamma^(c), rank V_n for each pair at each shell depth.
    Uses shell-only (non-cumulative) orbit vectors to control memory.

    ell_gamma is computed PER PAIR with that pair's own c (audit of
    2026-08-05: coherence_length previously ignored c entirely, so every pair
    plotted the identical curve and the reported "curve overlap confirms
    universality" was an artefact of the code, not an observation).
    """
    gens = build_generators(q)
    gens_arr = np.array(gens, dtype=np.int64)
    shells = bfs_shells(None, None, gens, q, bfs_frac)

    results = {}
    for (c, qmc) in pairs:
        ell_c = coherence_length(shells, q, c=c)
        depths, ells, ranks, rank_vc_list = [], [], [], []
        for depth in range(1, len(shells)):
            shell = shells[depth]
            if not shell:
                break
            shell_arr = np.array(shell, dtype=np.int64)
            x_c = fingerprint_vectors_batch(
                shell_arr, np.array([c, 0, 0]), gens_arr, q).T
            x_qc = fingerprint_vectors_batch(
                shell_arr, np.array([qmc, 0, 0]), gens_arr, q).T
            res = separator_scores(x_c, x_qc, tol=EPS_GS)
            depths.append(depth)
            ells.append(ell_c[depth])
            ranks.append(res.residual_rank)
            rank_vc_list.append(res.rank_vc)
        results[(c, qmc)] = dict(
            depths=depths, ells=ells, ranks=ranks, rank_vc=rank_vc_list)

    return results, shells, {c: coherence_length(shells, q, c=c) for (c, _) in pairs}


def make_figure(q: int, pairs: list[tuple[int, int]], results: dict,
                output_stem: str) -> None:
    colors     = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    linestyles = ['-', '--', '-.', ':']
    markers    = ['o', 's', '^', 'D']
    markevery  = 3

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax1, ax2, ax3 = axes

    for i, (c, qmc) in enumerate(pairs):
        d   = results[(c, qmc)]
        col = colors[i]
        ls  = linestyles[i]
        mk  = markers[i]
        lbl = "pair (%d,%d)" % (c, qmc)

        ax1.plot(d['depths'], d['ells'], color=col, lw=2, ls=ls,
                 marker=mk, markevery=markevery, ms=5, label=lbl)
        # small vertical jitter to reveal overlapping step curves
        jitter = i * 0.04
        ax2.step(d['depths'], [r + jitter for r in d['ranks']],
                 color=col, lw=2, ls=ls, where='post', label=lbl)
        ax3.plot(d['depths'], [r / q for r in d['rank_vc']],
                 color=col, lw=2, ls=ls, marker=mk, markevery=markevery, ms=5)

    ax1.axhline(0.1, color='red', ls='--', lw=1.2, alpha=0.7, label='eps = 0.1')
    ax1.set_ylabel('ell_gamma^(c)(n)', fontsize=12)
    ax1.set_title(
        'Fibre separator R_n vs per-pair coherence -- q=%d, %d conjugate pairs\n'
        'ell_gamma^(c) curves DIFFER across pairs (each computed with its own c); '
        'residual_rank stays 0 regardless'
        % (q, len(pairs)),
        fontsize=11)
    ax1.legend(fontsize=9, ncol=2)
    ax1.set_ylim(-0.05, 1.1)

    ax2.set_ylabel('residual_rank  (jitter added for visibility)', fontsize=11)
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_yticks([0, 1])
    ax2.legend(fontsize=9, ncol=2)

    ax3.axhline(1.0, color='grey', ls=':', lw=1.2, label='full saturation')
    ax3.set_ylabel('rank V_n / q', fontsize=12)
    ax3.set_xlabel('BFS depth n', fontsize=12)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = '%s_transition.%s' % (output_stem, ext)
        dpi = 150 if ext == 'png' else None
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
        print("Saved %s" % path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate fibre separator figure for Q1 paper.')
    parser.add_argument('--q', type=int, default=29,
                        help='Prime q (default: 29)')
    parser.add_argument('--bfs-frac', type=float, default=0.50,
                        help='BFS fraction of q^3 nodes (default: 0.50)')
    parser.add_argument('--output', default='separator_q29',
                        help='Output filename stem (without extension)')
    args = parser.parse_args()

    q = args.q
    # Default pairs: four conjugate pairs spread across (Z/qZ)^x
    pairs = [(1, q-1), (3, q-3), (7, q-7), (11, q-11)]
    # Ensure pairs are valid (c != 0, q-c != 0, c != q-c)
    pairs = [(c, qmc) for (c, qmc) in pairs
             if c > 0 and qmc > 0 and c != qmc][:4]

    print("q=%d, bfs_frac=%.2f, pairs=%s" % (q, args.bfs_frac, pairs))

    results, shells, ell = compute_separator_profile(q, pairs, args.bfs_frac)

    # Print summary table
    print()
    print("Summary: residual_rank at each depth")
    print("%6s %10s %8s" % ("depth", "ell_gamma", "resid_rk"))
    print("-" * 28)
    for depth, e, rw in zip(
            results[pairs[0]]['depths'],
            results[pairs[0]]['ells'],
            results[pairs[0]]['ranks']):
        print("%6d %10.4f %8d" % (depth, e, rw))

    make_figure(q, pairs, results, args.output)


if __name__ == '__main__':
    main()
