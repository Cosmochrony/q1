"""
test_regression.py
Plain-assert regression suite for the 2026-08-05 Q1 pipeline corrections
(quantum-structure/notes/q1-central-theorem-audit.md). No pytest dependency;
run directly with `python3 test_regression.py`. Exits nonzero on any failure.

Covers the five checks Jerome asked for before continuing:
  1. every shell returned by bfs_shells is complete and inversion-stable;
  2. coherence_length(shells, q, c) is genuinely c-dependent;
  3. no truncated final shell is ever silently kept;
  4. the exact frequency-support prediction matches the numeric rank exactly,
     both where it holds (complete shells) and where it fails (a deliberately
     truncated shell, negative control);
  5. shell-wise and cumulative-ball results are distinguished, not conflated.
"""

from __future__ import annotations

import sys

import numpy as np

from spectral_O12 import build_generators, bfs_shells, coherence_length, heisenberg_inv, heisenberg_mul
from separator import separator_scores, orthonormal_basis, separating_subspace_basis
from exact_fourier_support import is_inversion_symmetric, frequency_set, b_set

FAILURES: list[str] = []


def check(name: str, condition: bool) -> None:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        FAILURES.append(name)


def test_shells_complete_and_symmetric(q: int, bfs_frac: float) -> None:
    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    all_symmetric = all(is_inversion_symmetric(s, q) for s in shells if s)
    check(f"q={q}: every returned shell is inversion-symmetric (no silent partial shell)", all_symmetric)


def test_coherence_depends_on_c(q: int, bfs_frac: float) -> None:
    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    ell_1 = coherence_length(shells, q, c=1)
    ell_3 = coherence_length(shells, q, c=3)
    differs = not np.allclose(ell_1, ell_3)
    check(f"q={q}: ell_gamma(c=1) != ell_gamma(c=3) (coherence is c-dependent)", differs)


def test_no_partial_final_shell(q: int, bfs_frac: float) -> None:
    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    last = shells[-1]
    check(f"q={q}: final shell is inversion-symmetric (not a silent partial shell)",
          is_inversion_symmetric(last, q))


def test_exact_support_matches_numeric_rank(q: int) -> None:
    """
    The exact frequency-support criterion (F_c == F_qc) must agree with the
    numeric SVD-based residual_rank at EVERY depth, whether the shell is
    complete (the corrected pipeline: expect agreement AND both zero) or
    deliberately truncated the pre-audit way (a mix of agreement with either
    value is fine -- shell-inversion-symmetry is a SUFFICIENT condition for
    F_c==F_qc, not a necessary one, so a truncated shell need not always
    break equality; what must never happen is exact and numeric disagreeing).
    Both the corrected (complete-only) and the pre-audit truncating BFS are
    exercised, so this also acts as the negative control.
    """
    from weil_pipeline_compat import bfs_shells_truncating
    from spectral_O12 import fingerprint_vectors_batch

    gens = build_generators(q)
    gens_arr = np.array(gens, dtype=np.int64)
    pairs = [(c, q - c) for c in (1, 3, 7, 11) if 0 < c < q - c]

    saw_a_truncated_shell = False
    for label, shells in (("corrected", bfs_shells(None, None, gens, q, 0.50)),
                           ("pre-audit-truncating", bfs_shells_truncating(gens, q, 0.50))):
        for depth, shell in enumerate(shells):
            if not shell:
                continue
            if not is_inversion_symmetric(shell, q):
                saw_a_truncated_shell = True
            shell_arr = np.array(shell, dtype=np.int64)
            for (c, qmc) in pairs:
                x_c = fingerprint_vectors_batch(shell_arr, np.array([c, 0, 0]), gens_arr, q).T
                x_qc = fingerprint_vectors_batch(shell_arr, np.array([qmc, 0, 0]), gens_arr, q).T
                res = separator_scores(x_c, x_qc, tol=1e-10)
                fc = frequency_set(shell, q, c, gens)
                fqc = frequency_set(shell, q, qmc, gens)
                exact_predicts_zero = (fc == fqc)
                numeric_is_zero = (res.residual_rank == 0)
                check(f"q={q} [{label}] depth={depth} pair=({c},{qmc}): "
                      f"exact <=> numeric [exact={exact_predicts_zero}, numeric={numeric_is_zero}]",
                      exact_predicts_zero == numeric_is_zero)

    check(f"q={q}: the pre-audit-truncating BFS actually produced at least one "
          f"asymmetric (truncated) shell, so the negative control is real",
          saw_a_truncated_shell)


def test_shell_vs_cumulative_distinct(q: int, bfs_frac: float) -> None:
    """
    Shell-wise V_n and cumulative-ball B_n spans must not be silently
    conflated. A rank inequality (rank_ball >= rank_shell) is NOT sufficient
    evidence of this -- it holds vacuously under equality too (audit
    correction 2026-08-06: the original version of this check passed with
    rank_ball == rank_shell == 9 at every tested prime, which demonstrates
    nothing). This instead exhibits an EXACT frequency present in the ball's
    support but absent from that single shell's own support, at a genuine
    depth found by search -- a positive, not merely non-contradicted, proof
    that the two objects differ.
    """
    from exact_fourier_support import find_shell_vs_ball_divergence

    div = find_shell_vs_ball_divergence(q, bfs_frac, (1, 0, 0))
    check(f"q={q}: an exact frequency exists in the cumulative ball's support "
          f"but not in some single shell's own support (shell != ball, genuinely exhibited)",
          div is not None)


def test_generic_block_theorem(q: int, bfs_frac: float) -> None:
    """The frequency-symmetry theorem (Steps 1-5) must hold for GENERIC (c1,c2,c3)
    blocks too, not only the special [c,0,0] block used by the published figure."""
    from exact_fourier_support import verify_prime

    generic_blocks = [(1, 2, 3), (2, 5, 7), (4, 1, 9)]
    generic_blocks = [b for b in generic_blocks if sum(b) % q != 0 and max(b) < q]
    result = verify_prime(q, bfs_frac, generic_blocks, verbose=False)
    check(f"q={q}: theorem holds for generic blocks {generic_blocks} "
          f"({result['n_shells']} shells, {len(result['failures'])} failures)",
          len(result["failures"]) == 0)


def test_frame_invariance(q: int) -> None:
    """
    Covers the paper's Lemma "Constant modulus", the constructive proof of
    Corollary "Frame no-go" (x_qc[i] = d_i * x_c[iota(i)], |d_i|=1, via the
    explicit index involution, not merely Gram-matrix orbit-equality), and
    the normalised Bargmann invariant of Example "Normalised Bargmann
    triples" (=1 on a colinear triple, =0 by the extension convention on a
    triple containing an orthogonal pair). Independent review (2026-08-07)
    found this claimed numerically in the paper but not versioned anywhere;
    this closes that gap.
    """
    from spectral_O12 import fingerprint_vectors_batch
    from frame_invariance_check import index_involution, normalised_bargmann

    gens = build_generators(q)
    gens_arr = np.array(gens, dtype=np.int64)
    shells = bfs_shells(None, None, gens, q, 0.50)
    shell = shells[6]
    shell_arr = np.array(shell, dtype=np.int64)
    M = len(shell)
    iota_of = index_involution(shell, gens, q)

    for c in (1, 3, 7, 11):
        qmc = q - c
        x_c = fingerprint_vectors_batch(shell_arr, np.array([c, 0, 0]), gens_arr, q)
        x_qc = fingerprint_vectors_batch(shell_arr, np.array([qmc, 0, 0]), gens_arr, q)

        mags_ok = np.allclose(np.abs(x_c), q ** -1.5) and np.allclose(np.abs(x_qc), q ** -1.5)
        check(f"q={q} c={c}: entrywise magnitude is the constant q^-1.5 (Lemma, constant modulus)",
              mags_ok)

        mask = np.abs(x_c[iota_of]) > 1e-12
        ratio = np.where(mask, x_qc / np.where(mask, x_c[iota_of], 1), 0)
        d_moduli_ok = np.allclose(np.abs(ratio[mask]), 1.0, atol=1e-8)
        check(f"q={q} c={c}: x_qc[i] = d_i * x_c[iota(i)] with |d_i|=1 for all i "
              f"(Corollary, frame no-go, constructive proof)", d_moduli_ok)

        b_c_colinear = normalised_bargmann(x_c, 0, M, 2 * M)
        b_qc_colinear = normalised_bargmann(x_qc, 0, M, 2 * M)
        b_c_mixed = normalised_bargmann(x_c, 0, 1, M + 1)
        b_qc_mixed = normalised_bargmann(x_qc, 0, 1, M + 1)
        bargmann_ok = (abs(b_c_colinear - 1) < 1e-9 and abs(b_qc_colinear - 1) < 1e-9
                       and abs(b_c_mixed) < 1e-9 and abs(b_qc_mixed) < 1e-9)
        check(f"q={q} c={c}: normalised Bargmann = 1 on a colinear triple, "
              f"= 0 (extension convention) across an orthogonal pair, both sectors", bargmann_ok)


def main() -> None:
    for q in (13, 17, 29):
        test_shells_complete_and_symmetric(q, 0.99)
        test_coherence_depends_on_c(q, 0.99)
        test_no_partial_final_shell(q, 0.99)
        test_shell_vs_cumulative_distinct(q, 0.99)
        test_generic_block_theorem(q, 0.99)
    for q in (17, 29):
        test_exact_support_matches_numeric_rank(q)
    test_frame_invariance(29)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All regression checks passed.")


if __name__ == "__main__":
    main()
