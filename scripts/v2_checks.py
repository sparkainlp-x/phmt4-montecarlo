"""Numerical checks behind the README v2 claims (all outputs SYNTHETIC).

Run: python scripts/v2_checks.py
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phmt4.v2 as P  # noqa: E402

N = P.DEFAULT.N
rng = np.random.default_rng(0)

# 1. mirror: identity on pure states, sharpening on mixed states
pure_dev, gain = [], []
for _ in range(200):
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    r = P.pure_rho(psi)
    pure_dev.append(np.abs(P.crystallize(r, 0.58) - r).max())
    X = rng.normal(size=(N, 4)) + 1j * rng.normal(size=(N, 4))
    m = X @ X.conj().T
    m /= np.trace(m).real
    gain.append(P.purity(P.crystallize(m, 0.58)) - P.purity(m))
print("[1] mirror, strength 0.58, 200 states each")
print(f"    pure states : max |crystallize(rho) - rho| = {max(pure_dev):.1e} (identity: nothing to sharpen)")
print(f"    rank-4 mixed: purity gain mean {np.mean(gain):.3f}, min {np.min(gain):.3f}")

# 2. integrator: Strang splitting convergence (mirror off, one generation T = 0.4)
X = rng.normal(size=(N, 4)) + 1j * rng.normal(size=(N, 4))
rho0 = X @ X.conj().T
rho0 /= np.trace(rho0).real
T = 0.4


def integ(dt):
    cfg = replace(P.DEFAULT, dt=dt, steps_per_gen=int(round(T / dt)), mirror=False)
    return P.evolve(rho0[None], [3], cfg)[0]


ref = integ(T / 3200)
print("[2] Strang splitting, level 3, T = 0.4, error vs dt = T/3200 reference")
prev = None
for n in (20, 40, 80):
    e = np.abs(integ(T / n) - ref).max()
    print(f"    dt = T/{n:<3} max error {e:.2e}" + (f"  ratio {prev / e:.2f}" if prev else ""))
    prev = e

# 3. no power-iteration collapse: overlap with H's dominant eigenvector is conserved by U
E, V = np.linalg.eigh(P.build_H(3))
top = V[:, np.argmax(np.abs(E))]
psi = np.exp(1j * (0.1 * np.arange(N) + 0.22 * rng.normal(size=N)))
r = P.pure_rho(psi)[None]
cfg = replace(P.DEFAULT, dephasing=0.0, mirror=False)
ov0 = np.real(top.conj() @ r[0] @ top)
for _ in range(3):
    r = P.evolve(r, [3], cfg)
print(f"[3] overlap with dominant eigenvector of H: start {ov0:.4f}, after 60 exact steps "
      f"{np.real(top.conj() @ r[0] @ top):.4f} (v1 Euler step: 1.0000)")
