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


def check_constant_modulus(x: np.ndarray, q: int, label: str, tol: float = 1e-9) -> bool:
    """
    Each raw fingerprint vector psi_i(x) has entrywise modulus q^-1.5 (three
    factors of modulus q^-0.5 each, eq:weil-matrix-element / eq:fingerprint).
    Relative to the unit-norm Fourier basis vector e_K(x)=q^-0.5*exp(...),
    the scalar a_i = psi_i / e_K then has modulus q^-1 (Lemma "Constant
    modulus" in the paper) -- checked separately via check_frame_relation,
    which verifies |d_i|=1 for d_i = a_i^{-c}/a_{iota(i)}^{c} and so only
    needs the RATIO of moduli, not their absolute value. This function
    checks the more basic fact the ratio-based check assumes: |psi_i(x)| is
    the SAME constant q^-1.5 for every entry x and every index i. Returns
    True iff the deviation is within tolerance.
    """
    mags = np.abs(x)
    expected = q ** -1.5
    max_dev = np.max(np.abs(mags - expected))
    ok = max_dev < tol
    print(f"  [{label}] max |entrywise magnitude - q^-1.5| = {max_dev:.2e} "
          f"(expect ~0: every entry of every fingerprint vector has modulus q^-1.5, "
          f"hence a_i = psi_i/e_K has constant modulus q^-1) [{'OK' if ok else 'FAIL'}]")
    return ok


def check_frame_relation(x_c: np.ndarray, x_qc: np.ndarray, iota_of: np.ndarray,
                          tol: float = 1e-9) -> bool:
    """
    x_qc[i] should equal d_i * x_c[iota_of[i]] for a SINGLE scalar d_i with
    |d_i| = 1, for every i -- not merely |ratio(i,x)| = 1 at each coordinate
    x independently (a weaker, coordinate-wise-only check that a
    coordinate-dependent phase injected into x_qc, unrelated to a genuine
    constant d_i, would still pass: |d_i(x)| = 1 for each x does not imply
    d_i(x) is the same complex number for every x). The within-row spread
    check below is what actually tests single-scalar-ness; both this
    function and its caller must gate on it, not merely print it -- an
    earlier version of this script printed the deviations without ever
    returning them, so a genuine failure would not have affected the
    reported total (2026-08-07 review finding, fixed here). Returns True
    iff both the modulus and the spread are within tolerance.
    """
    lhs = x_qc
    rhs = x_c[iota_of]
    mask = np.abs(rhs) > 1e-12
    ratio = np.where(mask, lhs / np.where(mask, rhs, 1), 0)
    max_modulus_dev = 0.0
    max_within_row_spread = 0.0
    for i in range(x_c.shape[0]):
        row_ratios = ratio[i][mask[i]]
        if row_ratios.size == 0:
            continue
        max_modulus_dev = max(max_modulus_dev, np.max(np.abs(np.abs(row_ratios) - 1)))
        max_within_row_spread = max(max_within_row_spread, np.max(np.abs(row_ratios - row_ratios[0])))
    ok = max_modulus_dev < tol and max_within_row_spread < tol
    print(f"  max | |d_i| - 1 | over all i = {max_modulus_dev:.2e} (expect ~0: d_i in U(1))")
    print(f"  max within-row spread of d_i estimate = {max_within_row_spread:.2e} "
          f"(expect ~0: a SINGLE constant d_i per row, confirming x_qc[i] = d_i * x_c[iota(i)] "
          f"-- this is the check that actually distinguishes a genuine constant scalar from "
          f"coordinate-dependent phases that merely each have modulus 1) [{'OK' if ok else 'FAIL'}]")
    return ok


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
        if not check_constant_modulus(x_c, q, "c"):
            failures += 1
        if not check_constant_modulus(x_qc, q, "q-c"):
            failures += 1
        if not check_frame_relation(x_c, x_qc, iota_of):
            failures += 1

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

    negative_control_ok = check_frame_relation_rejects_coordinate_dependent_phase(q)
    if not negative_control_ok:
        failures += 1

    print(f"TOTAL failures: {failures}")
    if failures:
        raise SystemExit(1)


def check_frame_relation_rejects_coordinate_dependent_phase(q: int) -> bool:
    """
    Negative control (2026-08-07 review finding): check_frame_relation must
    REJECT a genuine violation of "a single constant d_i per row", not just
    accept anything with per-coordinate modulus 1. Injects a
    coordinate-dependent phase e^{i*2*pi*x/q} into one row of a synthetic
    x_qc array that otherwise satisfies x_qc = d * x_c pointwise -- every
    entry still has |ratio|=1, but the ratio is no longer a single constant
    across the row, reproducing exactly the review's own adversarial test
    (reported deviation ~1.997 there). check_frame_relation must return
    False on this input; returns True here iff it correctly does.
    """
    rng = np.random.default_rng(0)
    n, q_dim = 5, q
    x_c = (rng.normal(size=(n, q_dim)) + 1j * rng.normal(size=(n, q_dim)))
    x_c /= np.abs(x_c)  # unit-modulus entries, arbitrary phases (not pure Fourier lines)
    iota_of = np.arange(n)  # identity involution for this synthetic check

    # Genuine case: x_qc = single constant d_i * x_c[i], per row -- must PASS.
    d = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    x_qc_genuine = x_c * d[:, None]
    genuine_passes = check_frame_relation(x_c, x_qc_genuine, iota_of)

    # Attack: inject a coordinate-dependent phase into row 0 only -- must FAIL.
    x_qc_attack = x_qc_genuine.copy()
    coord_phase = np.exp(1j * 2 * np.pi * np.arange(q_dim) / q_dim)
    x_qc_attack[0] *= coord_phase
    attack_rejected = not check_frame_relation(x_c, x_qc_attack, iota_of)

    ok = genuine_passes and attack_rejected
    print(f"Negative control (coordinate-dependent-phase injection): "
          f"genuine case passes = {genuine_passes}, attack correctly rejected = "
          f"{attack_rejected} [{'OK' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    main()
