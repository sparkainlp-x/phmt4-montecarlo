"""Numerical checks behind the README findings for v1 (all outputs SYNTHETIC).

1. crystallize() on pure states: resonance(psi, crystallize(psi)).
2. Ablation: replace crystallize by the identity and rerun 40 samples, seed 42.
3. Euler + renormalize no-jump step vs exact unitary: overlap with the
   dominant eigenvector of H after k steps.

Run: python scripts/v1_checks.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phmt4.v1 as P  # noqa: E402

rng = np.random.default_rng(0)

# 1
res, dist = [], []
for s in (0.35, 0.40, 0.58, 0.78, 0.90):
    for _ in range(200):
        psi = P.normalize(rng.normal(size=P.N) + 1j * rng.normal(size=P.N))
        c = P.crystallize(psi, s)
        res.append(P.resonance(psi, c))
        dist.append(np.linalg.norm(c - psi))
print("[1] crystallize on 1000 random pure states (5 strengths x 200)")
print(f"    resonance(psi, crystallize(psi)): min {min(res):.16f}  max {max(res):.16f}")
print(f"    ||crystallize(psi) - psi||: mean {np.mean(dist):.3f}  max {np.max(dist):.3f}  (global phase only)")

# 2
def summary(label):
    g = np.random.default_rng(42)
    runs = [P.run_multigen(g) for _ in range(40)]
    keys = ("births_2", "births_3", "jumps", "mean_cf", "min_cf")
    print(f"    {label:<18}" + "  ".join(f"{k} {np.mean([r[k] for r in runs]):.6f}" for k in keys))

print("[2] ablation, 40 samples, seed 42")
summary("original mirror")
orig = P.crystallize
P.crystallize = lambda psi, strength=0.58: P.normalize(psi)
summary("identity mirror")
P.crystallize = orig

# 3
E, V = np.linalg.eigh(P.build_H(3))
top = V[:, np.argmax(np.abs(E))]
psi = P.normalize(np.exp(1j * (0.1 * np.arange(P.N) + 0.22 * rng.normal(size=P.N))))
He = P._cached_He(3, P.C.dephasing)
U = (V * np.exp(-1j * E * P.C.dt)) @ V.conj().T
print("[3] level 3, founder-like state; Cf(dominant eigenvector of H) =",
      f"{P.coherent_fraction(P.to_rho(top)):.4f}")
p, q = psi.copy(), psi.copy()
for k in range(1, 61):
    p = P.normalize(p - 1j * P.C.dt * (He @ p))
    q = U @ q
    if k in (1, 10, 20, 40, 60):
        print(f"    step {k:>2}: overlap with top eigvec  Euler {P.resonance(p, top):.4f}   exact {P.resonance(q, top):.4f}")
