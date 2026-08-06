"""
exact_fourier_support.py
Exact (integer-arithmetic, no floating-point tolerance) proof and verification
that conjugate Weil sectors generate IDENTICAL Fourier support -- V_n^(c) =
V_n^(-c) exactly, as subspaces, not merely isomorphic ones -- whenever the BFS
shell is complete (closed under group inversion). Generalised (2026-08-06,
Jerome's review) from the special block [c,0,0] to arbitrary triples
(c1,c2,c3); the special block is now the case (c,0,0) of the general theorem,
not a separate argument.

--------------------------------------------------------------------------
THE ARGUMENT, IN FULL, SO IT CAN BE READ WITHOUT RUNNING CODE
--------------------------------------------------------------------------

Step 1 -- every fingerprint vector is a pure Fourier mode, for ANY block.
fingerprint_vectors_batch builds, for a node u and generators (s1,s2,s3), the
product v1(x)*v2(x)*v3(x) where each vi is exp(2 pi i c_i (gamma_i + b_i(x -
a_i))/q)/sqrt(q) at ep_i = u.s1...s_i. As a function of x this product is a
pure mode e_K(x) = exp(2 pi i K x / q) times an x-independent scalar, with

    K(u; s1,s2,s3) = (c1+c2+c3) b_u + (c1+c2+c3) b_{s1} + (c2+c3) b_{s2}
                     + c3 b_{s3}   (mod q),

using that the b-coordinate of a group product is additive (b_out = b + b',
no cross terms) and b_i = b_u + b_{s1} + ... + b_{s_i}. Verified independently
against the actual numeric fingerprint vectors for a generic (non-special)
block: see test_regression.py's generic-block check.

Distinct-frequency modes are pairwise orthogonal, so the EXACT (not merely
dimension-matching) frequency SET F_{c1,c2,c3} = { K(u;s1,s2,s3) : u in shell,
s1,s2,s3 in gens } determines V_n^{(c1,c2,c3)} exactly, as a subspace.

Step 2 -- conjugation negates every component. rho_{(-c1,-c2,-c3)} =
conj(rho_{(c1,c2,c3)}) pointwise (each psi_{-ci} = conj(psi_ci)), so
F_{-c1,-c2,-c3} = -F_{c1,c2,c3} (mod q). Hence V_n^{-c} = V_n^{c} EXACTLY iff
F_c is closed under negation.

Step 3 -- complete BFS shells are inversion-symmetric (standard Cayley-graph
fact: {X,X^-1,Y,Y^-1} is closed under inversion, so graph distance from the
identity is inversion-invariant, so a COMPLETE sphere is closed under
u -> u^-1). heisenberg_inv negates a and b, so b_u -> -b_u is available.

Step 4 -- the generating-set b-values are themselves closed under negation:
{b_X, b_{X^-1}, b_Y, b_{Y^-1}} = {0, 0, 1, -1} = -{0,0,1,-1}. So for the
FULL index (u, s1, s2, s3), there is always a matching (u^-1, s1', s2', s3')
in the same combinatorial index set with b_{u^-1} = -b_u and each
b_{s_j'} = -b_{s_j} independently.

Step 5 -- K is LINEAR in (b_u, b_{s1}, b_{s2}, b_{s3}), so negating all four
inputs negates K exactly:

    K(u^-1; s1', s2', s3') = -K(u; s1, s2, s3)   (mod q),

for EVERY block (c1,c2,c3), not only the special one. Hence F_c is closed
under negation whenever the shell is complete, so V_n^{(c1,c2,c3)} =
V_n^{(-c1,-c2,-c3)} EXACTLY, for every prime q, every triple, every complete
shell -- independent of any coherence measure. The special block [c,0,0] is
the case c2=c3=0 of this, not a separate phenomenon.

This also repairs (not merely restates) the numeric residual_rank=0
observation: by rank-nullity (see separator.py's docstring), a residual_rank
of exactly 0 would prove V_c == V_qc rigorously once dim V_c == dim V_qc is
known (always true, conjugation preserves rank) -- the argument above
explains WHY residual_rank is exactly 0 (not merely numerically small)
analytically, for every q, block and n, rather than leaving it a case-by-case
floating-point observation. The numeric SVD code in separator.py only ever
reports a TOLERANCE zero; the exactness lives here, in integer arithmetic.
--------------------------------------------------------------------------

BRIDGE NOT YET CLOSED (Jerome's review, 2026-08-06): this file proves a
result about V_n^{(c)} := span{fingerprint_vectors_batch(...)}, the object
the k=3 pipeline actually computes. Q1's text defines V_n^{(c)} :=
span{rho_c(g) v0 : g in B_n} for an abstract orbit and an unspecified probe
v0. Whether the pipeline's k=3 fingerprint construction, its [c,0,0] block
convention, its uniform probe, and O17's v0 (a basis vector up to a phase)
are the SAME object as Q1's abstract V_n^{(c)} is a separate, unproved
identification -- this theorem explains the figure exactly; it does not, by
itself, repair Q1's abstract theorem statement until that bridge is typed.
"""

from __future__ import annotations

from itertools import product

from spectral_O12 import build_generators, heisenberg_inv, heisenberg_mul


def is_inversion_symmetric(shell: list[tuple[int, int, int]], q: int) -> bool:
    shell_set = set(shell)
    return all(heisenberg_inv(u, q) in shell_set for u in shell)


def frequency_general(u: tuple, s1: tuple, s2: tuple, s3: tuple, q: int, c_block: tuple) -> int:
    """Exact K(u; s1,s2,s3) for block (c1,c2,c3), integer arithmetic only."""
    c1, c2, c3 = c_block
    b_u, b1, b2, b3 = u[1], s1[1], s2[1], s3[1]
    return ((c1 + c2 + c3) * b_u + (c1 + c2 + c3) * b1 + (c2 + c3) * b2 + c3 * b3) % q


def frequency_set_general(shell: list[tuple[int, int, int]], q: int, c_block: tuple,
                           gens: list[tuple]) -> set[int]:
    """Exact frequency set F_{c_block} = { K(u;s1,s2,s3) : u in shell, s1,s2,s3 in gens }."""
    return {
        frequency_general(u, s1, s2, s3, q, c_block)
        for u in shell
        for s1, s2, s3 in product(gens, repeat=3)
    }


def frequency_set(shell: list[tuple[int, int, int]], q: int, c: int, gens: list[tuple]) -> set[int]:
    """Special block [c,0,0] -- kept as the c2=c3=0 case of frequency_set_general."""
    return frequency_set_general(shell, q, (c, 0, 0), gens)


def b_set(shell: list[tuple[int, int, int]], q: int) -> set[int]:
    return {b for (_a, b, _g) in shell}


def neg_block(c_block: tuple, q: int) -> tuple:
    return tuple((-c) % q for c in c_block)


def verify_prime(q: int, bfs_frac: float, blocks: list[tuple], verbose: bool = True) -> dict:
    """
    Build complete BFS shells, then check, with EXACT integer arithmetic, for
    every shell and every block (c1,c2,c3): if the shell is inversion-
    symmetric, F_c must equal F_{-c} (Steps 1-5 above). This tests the actual
    implication used by the proof, not a spurious biconditional with the
    weaker b-set symmetry (audit correction 2026-08-06: b-set symmetry can
    hold even when the shell itself is not fully inversion-symmetric, so
    "b_sym and not inv_sym" is not a real failure and must not be flagged).
    """
    from spectral_O12 import bfs_shells

    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)

    results = {"q": q, "n_shells": len(shells), "failures": []}
    for n, shell in enumerate(shells):
        if not shell:
            continue
        inv_sym = is_inversion_symmetric(shell, q)
        for c_block in blocks:
            fc = frequency_set_general(shell, q, c_block, gens)
            fnc = frequency_set_general(shell, q, neg_block(c_block, q), gens)
            neg_fc = {(-k) % q for k in fc}
            symmetric = (fc == fnc == neg_fc)
            # The only real implication the theorem makes: inv_sym => symmetric.
            # (The converse need not hold -- symmetric can happen "by accident"
            # on a non-inversion-symmetric shell; that is not a theorem failure.)
            if inv_sym and not symmetric:
                results["failures"].append(
                    dict(depth=n, c_block=c_block, inv_sym=inv_sym, symmetric=symmetric)
                )
                if verbose:
                    print(f"  [q={q}] THEOREM FAILURE at depth {n}, block {c_block}: "
                          f"inv_sym={inv_sym} but F_c != F_-c")

    if verbose:
        n_checked = results["n_shells"] * len(blocks)
        n_fail = len(results["failures"])
        print(f"q={q}: {results['n_shells']} shells, {len(blocks)} blocks, "
              f"{n_checked} (shell,block) checks, {n_fail} theorem failures "
              f"(inv_sym => F_c==F_-c violated)")
    return results


def find_shell_vs_ball_divergence(q: int, bfs_frac: float, c_block: tuple) -> dict | None:
    """
    Bug fix (Jerome's review, 2026-08-06): the earlier shell-vs-ball
    regression check only compared RANKS (rank_ball >= rank_shell), which
    can hold with equality and therefore never demonstrates that shell-only
    and cumulative-ball spans are actually different objects. This compares
    EXACT frequency sets and searches for a depth where the shell's own
    support is a STRICT subset of the cumulative ball's support up to that
    depth -- a genuine, exhibited divergence, not an inequality that can be
    vacuously satisfied by equality.
    """
    from spectral_O12 import bfs_shells

    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)

    ball_support: set[int] = set()
    for depth, shell in enumerate(shells):
        if not shell:
            continue
        shell_support = frequency_set_general(shell, q, c_block, gens)
        new_ball_support = ball_support | shell_support
        if depth > 0 and shell_support < new_ball_support:
            # shell_support is a STRICT subset of the ball support at this depth:
            # the ball sees frequencies the shell alone does not.
            return dict(depth=depth, shell_support=sorted(shell_support),
                        ball_support=sorted(new_ball_support),
                        only_in_ball=sorted(new_ball_support - shell_support))
        ball_support = new_ball_support
    return None


def main() -> None:
    import argparse
    import random

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primes", default="13,17,29,37,53,61")
    parser.add_argument("--bfs-frac", type=float, default=0.99)
    args = parser.parse_args()

    rng = random.Random(0)
    all_results = []
    for q in [int(p) for p in args.primes.split(",")]:
        special_blocks = [(c, 0, 0) for c in (1, 3, 7, 11) if 0 < c < q - c]
        generic_blocks = []
        for _ in range(4):
            c1, c2, c3 = (rng.randrange(1, q) for _ in range(3))
            if (c1 + c2 + c3) % q != 0:
                generic_blocks.append((c1, c2, c3))
        all_results.append(verify_prime(q, args.bfs_frac, special_blocks + generic_blocks))

        div = find_shell_vs_ball_divergence(q, args.bfs_frac, (1, 0, 0))
        if div is not None:
            print(f"  [q={q}] shell/ball divergence found at depth {div['depth']}: "
                  f"{len(div['only_in_ball'])} frequencies present in the ball but not the shell alone")
        else:
            print(f"  [q={q}] no shell/ball divergence found up to bfs_frac={args.bfs_frac} "
                  f"(shell support already saturates the ball's at every tested depth)")

    total_failures = sum(len(r["failures"]) for r in all_results)
    print()
    print(f"TOTAL theorem failures (inv_sym but F_c != F_-c) across all primes/shells/blocks: {total_failures}")
    print("0 means the theorem (Steps 1-5 above) holds exactly on every complete shell tested, "
          "for both the special block and generic (c1,c2,c3) triples, with no floating-point "
          "tolerance involved anywhere.")


if __name__ == "__main__":
    main()
