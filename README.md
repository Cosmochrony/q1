# Q1 — From Admissibility to Quantum Structure

This repository contains the source of the **Q1 Cosmochrony paper**
[*From Admissibility to Quantum Structure: Phase Coherence and Correlations from
Non-Injective Projection*](out/Q1-paper.pdf).

This paper establishes that quantum-mechanical **phase coherence is not an independent
postulate but a structural consequence of projective admissibility**.

## Core Result

The central theorem shows that any admissible transition preserves the Born–Infeld
indiscernibility of conjugate Weil blocks $\rho_c$ and $\rho_{q-c}$ on the Heisenberg Cayley
graph $\mathrm{Cay}(\mathrm{Heis}_3(\mathbb{Z}/q), S_q)$, and thereby maintains the metaplectic
phase coherence of the admissible fibre. The proof rests on three independently established
results: the minimal parity fibre structure of the non-injective projection (O18), the
dynamical identity of conjugate Weil blocks (O17), and the discrete admissibility filter of
projection locking (O22).

A constructive separating functional $R_n^{(c)}$ is exhibited and proved BI-admissible.
Numerical validation for $q = 29$ across four conjugate pairs confirms
$\operatorname{rank} W_n^{(c)} = 0$ exactly throughout the admissible regime — establishing
structural indistinguishability of conjugate blocks as a representation-theoretic fact rather
than a dynamical coincidence. The admissible fibre therefore carries complex amplitude
structure with interference, the natural precursor of an effective Hilbert sector — without
quantum mechanics as a postulate.

## Keywords

Phase coherence, non-injective projection, Weil blocks, Born–Infeld indiscernibility,
metaplectic phase, Heisenberg Cayley graph, emergent Hilbert space.

## Repository Contents

```
q1/
├── code/        # Numerical validation scripts (+ requirements.txt)
├── tex/         # LaTeX sources (main + cosmochrony-bibliography.bib)
├── out/         # Compiled paper PDF (Q1-paper.pdf)
├── zenodo.json  # Zenodo deposition metadata
└── README.md
```

## Links

- 🔗 DOI: [10.5281/zenodo.19561060](https://doi.org/10.5281/zenodo.19561060)
- 🌐 Website: https://cosmochrony.org/science/quantum-structure/q1/

## Citation

> J. Beau, *From Admissibility to Quantum Structure: Phase Coherence and Correlations from
> Non-Injective Projection*, Zenodo, 2026. DOI: 10.5281/zenodo.19561060.

## Acknowledgements

Portions of the editorial refinement benefited from iterative interactions with large
language models, used as analytical assistants. All claims and final formulations remain
the sole responsibility of the author.
