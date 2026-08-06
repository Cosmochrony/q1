from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class SeparatorResult:
    rank_vc: int
    rank_vqc: int
    residual_rank: int
    energy_c: float
    energy_qc: float
    r_trace: float | None


def _matrix_rank_from_svals(svals: Array, tol: float) -> int:
    return int(np.sum(svals > tol))


def orthonormal_basis(x: Array, tol: float = 1e-10) -> Array:
    """
    Return an orthonormal basis for the column span of x.

    Parameters
    ----------
    x : array, shape (d, m)
        Columns are vectors spanning the target subspace.
    tol : float
        Singular-value cutoff.

    Returns
    -------
    q : array, shape (d, r)
        Orthonormal basis of the span of x.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    if x.size == 0:
        return np.zeros((x.shape[0], 0), dtype=np.complex128)

    u, s, _vh = np.linalg.svd(x, full_matrices=False)
    r = _matrix_rank_from_svals(s, tol)
    if r == 0:
        return np.zeros((x.shape[0], 0), dtype=np.complex128)
    return u[:, :r].astype(np.complex128, copy=False)


def projector_from_basis(q: Array) -> Array:
    """
    Build the orthogonal projector P = Q Q^* from an orthonormal basis Q.
    """
    if q.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {q.shape}")
    return q @ q.conj().T


def separating_subspace_basis(qc: Array, qqc: Array, tol: float = 1e-10) -> Array:
    """
    Compute an orthonormal basis of the RESIDUAL (I - P_qc) V_c, i.e. the
    orthogonal projection of V_c onto (V_qc)^perp.

    This is NOT, in general, a basis of the set intersection V_c ∩ (V_qc)^perp
    (audit of 2026-08-05: two distinct, non-orthogonal 1-dimensional subspaces
    give a nonzero residual here while their true intersection is {0} —
    confirmed by an explicit countermodel). Do not call this "W" or its rank
    "rank_W" as if it computed the intersection; that identification is what
    the published Q1 text and this docstring both got wrong.

    What it DOES prove, exactly, by rank-nullity applied to v -> (I-P_qc)v
    restricted to V_c (kernel = V_c ∩ V_qc, since ker(I-P_qc) = V_qc):

        dim V_c = dim(V_c ∩ V_qc) + residual_rank.

    So residual_rank == 0  <=>  V_c ∩ V_qc = V_c  <=>  V_c subseteq V_qc.
    Combined with dim V_c == dim V_qc (conjugation preserves rank, always,
    for ANY c -- elementary, not specific to this construction), a TRUE
    residual_rank == 0 would prove V_c == V_qc exactly, as literal subspaces,
    not merely isomorphic ones. This is a valid route to the subspace-identity
    conclusion Q1 wanted; the invalid route was claiming the raw intersection
    is automatically nonzero whenever V_c != V_qc, which is false in general.

    CAVEAT: this function itself only ever reports a SVD-tolerance zero
    (`tol`), not a mathematically exact one -- "residual_rank == 0" as
    returned by THIS code is numerical evidence, not a proof. The exactness
    comes only from the separate combinatorial argument in
    exact_fourier_support.py (integer arithmetic, no tolerance), which is what
    actually establishes V_c == V_qc for the special block [c,0,0]; do not
    read a numeric zero from this module, on its own, as already exact.
    """
    if qc.shape[0] != qqc.shape[0]:
        raise ValueError(
            f"Dimension mismatch: qc has ambient dim {qc.shape[0]}, "
            f"qqc has ambient dim {qqc.shape[0]}"
        )

    p_qc = projector_from_basis(qqc)
    residual = qc - p_qc @ qc
    return orthonormal_basis(residual, tol=tol)


def empirical_covariance(x: Array) -> Array:
    """
    Empirical covariance/operator C = (1/m) X X^* for column vectors in X.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    m = x.shape[1]
    if m == 0:
        raise ValueError("Cannot build covariance from zero vectors.")
    return (x @ x.conj().T) / float(m)


def trace_separator_energy(qw: Array, c_n: Array) -> float:
    """
    Compute Tr(P_W C_n) without materializing P_W if possible.
    """
    if c_n.ndim != 2 or c_n.shape[0] != c_n.shape[1]:
        raise ValueError(f"C_n must be square, got shape {c_n.shape}")
    if qw.shape[0] != c_n.shape[0]:
        raise ValueError(
            f"Ambient dimension mismatch: qw has {qw.shape[0]}, C_n has {c_n.shape[0]}"
        )
    if qw.shape[1] == 0:
        return 0.0
    # Tr(Q Q^* C) = Tr(Q^* C Q)
    return float(np.real(np.trace(qw.conj().T @ c_n @ qw)))


def separator_scores(
    x_c: Array,
    x_qc: Array,
    c_n: Array | None = None,
    tol: float = 1e-10,
) -> SeparatorResult:
    """
    Compute the separator W and associated energies.

    Parameters
    ----------
    x_c, x_qc : array, shape (d, m_c), (d, m_qc)
        Columns are orbit vectors for blocks c and q-c.
    c_n : optional square array, shape (d, d)
        Covariance/operator to probe with Tr(P_W C_n). If omitted, the script
        computes only direct projected energies on x_c and x_qc.
    tol : float
        Numerical cutoff for rank decisions.

    Returns
    -------
    SeparatorResult
    """
    qc = orthonormal_basis(x_c, tol=tol)
    qqc = orthonormal_basis(x_qc, tol=tol)
    qw = separating_subspace_basis(qc, qqc, tol=tol)

    p_w = projector_from_basis(qw)

    energy_c = float(np.real(np.linalg.norm(p_w @ x_c, ord="fro") ** 2))
    energy_qc = float(np.real(np.linalg.norm(p_w @ x_qc, ord="fro") ** 2))

    r_trace = None
    if c_n is not None:
        r_trace = trace_separator_energy(qw, c_n)

    return SeparatorResult(
        rank_vc=qc.shape[1],
        rank_vqc=qqc.shape[1],
        residual_rank=qw.shape[1],
        energy_c=energy_c,
        energy_qc=energy_qc,
        r_trace=r_trace,
    )


def _load_array(npz: Any, key: str) -> Array:
    if key not in npz:
        raise KeyError(
            f"Missing key '{key}' in NPZ file. "
            f"Available keys: {sorted(npz.files)}"
        )
    arr = np.asarray(npz[key])
    return arr.astype(np.complex128, copy=False)


def _maybe_load_covariance(npz: Any, key: str | None) -> Array | None:
    if key is None:
        return None
    return _load_array(npz, key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the Gram-Schmidt separator R_n^(c)."
    )
    parser.add_argument("npz", type=Path, help="NPZ file containing orbit vectors")
    parser.add_argument(
        "--x-c-key",
        default="X_c",
        help="NPZ key for orbit vectors of block c, shape (d, m_c)",
    )
    parser.add_argument(
        "--x-qc-key",
        default="X_qc",
        help="NPZ key for orbit vectors of block q-c, shape (d, m_qc)",
    )
    parser.add_argument(
        "--cov-key",
        default=None,
        help="Optional NPZ key for covariance/operator C_n, shape (d, d)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Numerical tolerance for rank decisions",
    )
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as npz:
        x_c = _load_array(npz, args.x_c_key)
        x_qc = _load_array(npz, args.x_qc_key)
        c_n = _maybe_load_covariance(npz, args.cov_key)

    result = separator_scores(x_c=x_c, x_qc=x_qc, c_n=c_n, tol=args.tol)

    payload = {
        "rank_vc": result.rank_vc,
        "rank_vqc": result.rank_vqc,
        "residual_rank": result.residual_rank,
        "energy_c": result.energy_c,
        "energy_qc": result.energy_qc,
        "r_trace": result.r_trace,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()