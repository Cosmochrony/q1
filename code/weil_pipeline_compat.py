"""
weil_pipeline_compat.py
Reproduces the PRE-AUDIT (2026-08-05) mid-shell-truncating BFS, kept only as a
negative control for test_regression.py -- to confirm the exact Fourier-
support criterion correctly predicts FAILURE on a genuinely truncated shell,
not just success on complete ones. Do not use this for any real computation;
bfs_shells in spectral_O12.py is the corrected version.
"""

from spectral_O12 import heisenberg_mul


def bfs_shells_truncating(gens: list[tuple], q: int, max_fraction: float) -> list[list[tuple]]:
    identity = (0, 0, 0)
    max_nodes = int(max_fraction * q ** 3)
    visited = {identity}
    current_shell = [identity]
    shells = [current_shell]
    total = 1
    while total < max_nodes:
        next_shell = []
        for u in current_shell:
            for g in gens:
                v = heisenberg_mul(u, g, q)
                if v not in visited:
                    visited.add(v)
                    next_shell.append(v)
                    total += 1
                    if total >= max_nodes:
                        break
            if total >= max_nodes:
                break
        if not next_shell:
            break
        shells.append(next_shell)
        current_shell = next_shell
    return shells
