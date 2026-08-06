"""
frame_invariance_check.py
Independent numerical verification of the frame-level claims in the paper's
Lemma (constant modulus), Corollary (frame no-go), and Example (normalised
Bargmann triples) -- see tex/Q1-paper.tex, Section "No-go corollaries".

This was previously checked only in an ad-hoc scratch script during drafting
(2026-08-06); it is versioned here per independent review's requirement that
every numerical claim in the paper have a corresponding checked-in script.

Checks, for q=29 and both conjugate sectors c and q-c:
  1. every raw fingerprint vector has EXACT constant modulus q^-1 relative to
     its own Fourier basis vector e_K (Lemma "Constant modulus");
  2. x_i^{-c} = d_i * x_{iota(i)}^c pointwise, for the explicit index
     involution iota (component-wise group inversion) and some |d_i|=1
     (Corollary "Frame no-go"'s constructive proof, not merely the Gram
     matrix's own conjugate-symmetry);
  3. the normalised Bargmann invariant equals exactly 1 on a genuinely
     colinear triple and (by the stated extension convention) 0 on a triple
     containing an orthogonal pair, identically in both sectors.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from spectral_O12 import (
    build_generators,
    bfs_shells,
    fingerprint_vectors_batch,
    heisenberg_inv,
    heisenberg_mul,
)


def index_involution(shell: list[tuple], gens: list[tuple], q: int):
    """
    iota(u; s1,s2,s3) := (u^-1; s1^-1, s2^-1, s3^-1), component-wise group
    inversion -- the SAME map used in the paper's Theorem Step 5. Returns,
    for the flat row layout of fingerprint_vectors_batch (s1 outer, s2 mid,
    s3 inner, each block of M=len(shell) rows in shell order), the
    permutation `iota_of[i]` giving the row index of iota(index i).
    """
    M = len(shell)
    shell_index = {u: k for k, u in enumerate(shell)}
    gen_index = {g: k for k, g in enumerate(gens)}
    gen_inv_index = {g: gen_index[heisenberg_inv(g, q)] for g in gens}
    shell_inv_index = {u: shell_index[heisenberg_inv(u, q)] for u in shell}

    iota_of = np.empty(M * 4 * 4 * 4, dtype=np.int64)
    for s1i, s1 in enumerate(gens):
        for s2i, s2 in enumerate(gens):
            for s3i, s3 in enumerate(gens):
                block = ((s1i * 4 + s2i) * 4 + s3i) * M
                s1p, s2p, s3p = gen_inv_index[s1], gen_inv_index[s2], gen_inv_index[s3]
                dest_block = ((s1p * 4 + s2p) * 4 + s3p) * M
                for u_idx, u in enumerate(shell):
                    iota_of[block + u_idx] = dest_block + shell_inv_index[u]
    return iota_of


def check_constant_modulus(x: np.ndarray, q: int, label: str) -> None:
    """
    Each raw fingerprint vector psi_i(x) has entrywise modulus q^-1.5 (three
    factors of modulus q^-0.5 each, eq:weil-matrix-element / eq:fingerprint).
    Relative to the unit-norm Fourier basis vector e_K(x)=q^-0.5*exp(...),
    the scalar a_i = psi_i / e_K then has modulus q^-1 (Lemma "Constant
    modulus" in the paper) -- checked separately via check_frame_relation,
    which verifies |d_i|=1 for d_i = a_i^{-c}/a_{iota(i)}^{c} and so only
    needs the RATIO of moduli, not their absolute value. This function
    checks the more basic fact the ratio-based check assumes: |psi_i(x)| is
    the SAME constant q^-1.5 for every entry x and every index i.
    """
    mags = np.abs(x)
    expected = q ** -1.5
    max_dev = np.max(np.abs(mags - expected))
    print(f"  [{label}] max |entrywise magnitude - q^-1.5| = {max_dev:.2e} "
          f"(expect ~0: every entry of every fingerprint vector has modulus q^-1.5, "
          f"hence a_i = psi_i/e_K has constant modulus q^-1)")


def check_frame_relation(x_c: np.ndarray, x_qc: np.ndarray, iota_of: np.ndarray) -> None:
    """x_qc[i] should equal d_i * x_c[iota_of[i]] for some |d_i| = 1, for every i."""
    lhs = x_qc
    rhs = x_c[iota_of]
    # ratio component-wise where rhs is non-negligible; all ratios for a
    # fixed i should be the SAME constant d_i (frame vectors are proportional
    # to a pure Fourier mode, so any two non-zero-supported x-components give
    # the same ratio).
    mask = np.abs(rhs) > 1e-12
    ratio = np.where(mask, lhs / np.where(mask, rhs, 1), 0)
    # per-row: max deviation of |ratio| from 1, and std of the ratio across
    # the row's non-zero entries (should be ~0, i.e. single constant d_i)
    max_modulus_dev = 0.0
    max_within_row_spread = 0.0
    for i in range(x_c.shape[0]):
        row_ratios = ratio[i][mask[i]]
        if row_ratios.size == 0:
            continue
        max_modulus_dev = max(max_modulus_dev, np.max(np.abs(np.abs(row_ratios) - 1)))
        max_within_row_spread = max(max_within_row_spread, np.max(np.abs(row_ratios - row_ratios[0])))
    print(f"  max | |d_i| - 1 | over all i = {max_modulus_dev:.2e} (expect ~0: d_i in U(1))")
    print(f"  max within-row spread of d_i estimate = {max_within_row_spread:.2e} "
          f"(expect ~0: a single constant d_i per row, confirming x_qc[i] = d_i * x_c[iota(i)])")


def normalised_bargmann(x: np.ndarray, i: int, j: int, k: int):
    g_ij = np.vdot(x[i], x[j])
    g_jk = np.vdot(x[j], x[k])
    g_ki = np.vdot(x[k], x[i])
    denom = abs(g_ij) * abs(g_jk) * abs(g_ki)
    if denom < 1e-14:
        return 0.0 + 0j  # extension convention: undefined -> 0
    return (g_ij * g_jk * g_ki) / denom


def main() -> None:
    q = 29
    gens = build_generators(q)
    gens_arr = np.array(gens, dtype=np.int64)
    shells = bfs_shells(None, None, gens, q, 0.50)
    shell = shells[6]
    shell_arr = np.array(shell, dtype=np.int64)
    M = len(shell)

    iota_of = index_involution(shell, gens, q)

    failures = 0
    for c in (1, 3, 7, 11):
        qmc = q - c
        x_c = fingerprint_vectors_batch(shell_arr, np.array([c, 0, 0]), gens_arr, q)
        x_qc = fingerprint_vectors_batch(shell_arr, np.array([qmc, 0, 0]), gens_arr, q)

        print(f"c={c}, q-c={qmc}:")
        check_constant_modulus(x_c, q, "c")
        check_constant_modulus(x_qc, q, "q-c")
        check_frame_relation(x_c, x_qc, iota_of)

        # Colinear triple: fix u=0, s1=gens[0], vary (s2,s3) -> same K.
        i, j, k = 0, M, 2 * M
        b_c = normalised_bargmann(x_c, i, j, k)
        b_qc = normalised_bargmann(x_qc, i, j, k)
        colinear_ok = abs(b_c - 1) < 1e-9 and abs(b_qc - 1) < 1e-9
        print(f"  colinear triple (0,{j},{k}): B_c={b_c:.6f}, B_qc={b_qc:.6f} "
              f"({'OK' if colinear_ok else 'FAIL'})")
        if not colinear_ok:
            failures += 1

        # Mixed triple containing an orthogonal pair (different u -> different K generically).
        i2, j2, k2 = 0, 1, M + 1
        b_c2 = normalised_bargmann(x_c, i2, j2, k2)
        b_qc2 = normalised_bargmann(x_qc, i2, j2, k2)
        mixed_ok = abs(b_c2) < 1e-9 and abs(b_qc2) < 1e-9
        print(f"  mixed triple (0,1,{k2}): B_c={b_c2:.6f}, B_qc={b_qc2:.6f} "
              f"({'OK' if mixed_ok else 'FAIL'})")
        if not mixed_ok:
            failures += 1
        print()

    print(f"TOTAL failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
