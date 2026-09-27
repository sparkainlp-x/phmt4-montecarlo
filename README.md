# phmt4-montecarlo

[![CI](https://github.com/sparkainlp-x/phmt4-montecarlo/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkainlp-x/phmt4-montecarlo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Outputs: SYNTHETIC](https://img.shields.io/badge/outputs-SYNTHETIC-lightgrey.svg)](#what-it-is-not)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22999270.svg)](https://doi.org/10.5281/zenodo.22999270)

PHMT-4 Monte Carlo: **a heuristic, classical numpy simulation**. It has 32-dimensional
"membranes", dephasing evolution, an eigenvalue-boost "crystallization mirror",
binary/ternary births gated by resonance thresholds, an Observer that reports the coherent
fraction Cf, and a Calibrator that adapts parameters.

This branch holds **v2** (`phmt4/v2.py`), a density-matrix redesign that fixes the design
flaws found in the original prototype. The faithful original with minimal fixes (v1) and its
checks are preserved at tag [`v1-original`](https://github.com/sparkainlp-x/phmt4-montecarlo/tree/v1-original).

## What it is NOT

- **Not a consciousness theory.** Nothing here measures or models consciousness.
- **Not quantum hardware and not a quantum computer.** It is ordinary floating-point linear algebra on a CPU. "Density matrix", "Lindblad" and "dephasing" name the math it uses, not a device.
- **Not biology or medicine.** "Membrane", "birth" and "fecundity" are metaphors for matrices and population rules.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python phmt4_montecarlo.py --samples 40 --seed 42 --csv out.csv --json out.json   # or: python -m phmt4
python phmt4_montecarlo.py --samples 40 --seed 42 --no-mirror                     # ablation
python -m pytest              # 18 tests, a few seconds
python scripts/v2_checks.py   # integrator / mirror checks
python scripts/sweep.py       # parameter sweep -> results/sweep_seed42.csv (~1 min on 8 cores)
```

## v1 vs v2: side-by-side (40 samples, seed 42), all **SYNTHETIC**

Machine: 8-core Linux box, Python 3.13.5, numpy 2.5.3, pandas 3.0.6. **v1 numbers come from tag
[`v1-original`](https://github.com/sparkainlp-x/phmt4-montecarlo/tree/v1-original)** ([results/v1_seed42.txt](https://github.com/sparkainlp-x/phmt4-montecarlo/tree/v1-original/results/v1_seed42.txt),
[results/v1_checks.txt](https://github.com/sparkainlp-x/phmt4-montecarlo/tree/v1-original/results/v1_checks.txt)). v2 numbers come from
[`results/`](results/) on this branch.

| Metric (mean over 40 runs) | v1 (tag v1-original) | v2 | v2, mirror off (ablation) | Tag |
|---|---|---|---|---|
| Final population | 48.00 | 48.00 | 48.00 | SYNTHETIC |
| Binary / ternary births | 28.25 / 13.75 | 27.52 / 14.47 | 28.15 / 13.85 | SYNTHETIC |
| Final Cf (mean / std / min) | 0.971 / 0.025 / 0.808 | 0.817 / 0.065 / 0.710 | 0.769 / 0.064 / 0.657 | SYNTHETIC |
| Founder / binary / ternary Cf | 0.960 / 0.971 / 0.970 | 0.806 / 0.818 / 0.817 | 0.772 / 0.769 / 0.769 | SYNTHETIC |
| Mean purity tr ρ² (min) | n/a (pure states) | 0.9999 (0.9992) | 0.8754 (0.8285) | SYNTHETIC |
| Entropy S/ln N | n/a | 0.0002 | 0.1241 | SYNTHETIC |
| Post-birth mirror purity gain | 0 (no-op, see below) | 0.0331 per birth | 0 | SYNTHETIC |
| Effect of removing the mirror | births identical, Cf −0.00007 | purity −0.125, Cf −0.048 | n/a | SYNTHETIC |
| Max \|tr ρ − 1\| / Hermiticity error / min eigenvalue | n/a | 1.8e-15 / 0 / −5.6e-16 | 1.2e-13 / 0 / 6.8e-04 | SYNTHETIC |
| Jumps | 0.57 | n/a (ensemble average) | n/a | SYNTHETIC |
| Protection held | 100.0% | 100.0% | 100.0% | SYNTHETIC |
| Generations completed (of 6) | 2.1 | 2.4 | 2.5 | SYNTHETIC |
| Runtime, 40 samples | 10.44 s original script; 6.69 s v1 with caching | 4.33 s | 1.68 s | SYNTHETIC |

How to read this. The table shows three measured differences, and **nothing here says v2 is
"better" as a model**:
1. **Correctness.** v2 keeps density matrices valid to machine precision, and its integrator converges at second order (below). v1's integrator does not preserve the dynamics.
2. **The mirror has a measurable effect** in v2. In v1 it did nothing.
3. **Speed.** v2 is 4.33 s vs 10.44 s for the original script on this machine.

v1's higher Cf (0.97) is mostly an integrator artifact (finding 2 below), so the lower v2 Cf
is **not** a regression and not an improvement. It is what this Hamiltonian gives without the artifact.

## Why v2 (findings on v1, verified at tag v1-original)

1. **The crystallization mirror was a no-op on pure states.** For ρ = |ψ⟩⟨ψ| (rank 1), the eigenvalue boost keeps the same top eigenvector. Over 1000 random states, `resonance(psi, crystallize(psi))` fell between 0.9999999999999993 and 1.0000000000000009, so only a global phase changed. Swapping in the identity gave identical births (28.25 / 13.75) and moved mean Cf from 0.970738 to 0.970665.
2. **The first-order non-unitary step acted as power iteration.** `psi - i dt H_eff psi` plus renormalization amplifies high-|E| components (E up to about 31, dt = 0.02). A founder's overlap with H's dominant eigenvector went from 0.44 to 0.994 in one generation (20 steps). Every membrane collapsed toward that eigenvector, whose Cf is 0.9753, which explains v1's uniform Cf of about 0.97.
3. Stochastic jumps projected onto a single basis state (Cf = 0).
4. The shared module-level RNG and the eigenvector phase from LAPACK made results depend on call order and platform. The RNG was fixed in v1. The phase issue disappears in v2, because density matrices carry no phase.

## v2 design, in plain words

- **Each membrane is a density matrix** (32×32, Hermitian, positive, trace 1), so dephasing can make it genuinely mixed. That gives the mirror something to act on.
- **Evolution is the Lindblad equation** with the same H(level) and the same dephasing operators L_i = sqrt(γ)|i⟩⟨i| as v1. It is integrated by Strang splitting: half a step of exact dephasing (off-diagonals × e^(−γ dt/2)), one exact unitary step e^(−iH dt) from the eigendecomposition of H, then another half dephasing step. Each piece is a valid quantum channel (CPTP), so trace, Hermiticity and positivity hold by construction. v1's random jumps are replaced by their average, which removes jump noise and the jump counter.
- **Mirror:** ρ → ρ^g / tr ρ^g with g = 1 + 2·strength. It leaves pure states unchanged (correct: nothing to sharpen), and it raises the purity of mixed states (mean +0.083 on random rank-4 states at strength 0.58, SYNTHETIC).
- **Resonance** is the Uhlmann fidelity. It equals v1's \|⟨a\|b⟩\|² on pure states (tested).
- **Births** are convex mixtures of the parents' density matrices, followed by the post-birth mirror. A superposition of mixed states is not well defined, but a mixture is. Because of this, the Calibrator's "strengthen crystallization mirror" action now changes the states.
- Everything else is unchanged from v1: thresholds, fecundity, mixing weights, cap, Calibrator rules and bounds, founder recipe, and Cf (normalized l1 coherence).
- **Speed:** all membranes evolve in one batched numpy pass, U(level) and the dephasing mask are cached, and the fidelity matrix is batched.
- **Output:** a seeded CLI (`--dephasing`, `--cap`, `--thresh2`, `--thresh3`, `--no-mirror`) with `--csv` (per-sample rows) and `--json` (means, runtime, config), both tagged SYNTHETIC.

### Integrator check (SYNTHETIC, [results/v2_checks.txt](results/v2_checks.txt))

| dt | max error vs dt = T/3200 | ratio |
|---|---|---|
| T/20 (default, dt = 0.02) | 4.43e-07 | – |
| T/40 | 1.11e-07 | 4.01 |
| T/80 | 2.76e-08 | 4.00 |

A ratio of 4 means second-order convergence. Without dephasing, the overlap with H's dominant
eigenvector stays at 0.4114 after 60 exact steps (v1's Euler step drives it to 1.0000).

## Parameter sweep (SYNTHETIC)

Question: is there a regime where the population does **not** simply hit the cap by
generation 2–3, and where the mirror's effect on purity and Cf is visible? Every setting
runs 40 samples at seed 42 (no sample reduction was needed). Reproduce with:

```bash
python scripts/sweep.py                                           # grid 1 -> results/sweep_seed42.csv
python scripts/sweep.py --thresh2 0.80 0.85 0.88 0.90 --dephasing 0.05 0.5 2.0 --caps 96 \
    --out results/sweep_thresholds_seed42.csv                     # grid 2 (follow-up)
python -m phmt4 --dephasing 0.5 --cap 96 --thresh2 0.90 --thresh3 0.92   # one setting
```

Column notes:
- "gens" is the mean number of generations completed.
- "mirror gain" is the mean purity increase from the post-birth mirror.
- Runtimes are **single-threaded** wall times (7 settings run in parallel, 1 BLAS thread each), so they are not comparable with the multi-threaded runtimes above.
- Binary births are under 0.5 per membrane whenever the cap binds, because births are cut off at the cap.

### Grid 1: dephasing × cap × mirror (default thresholds 0.70 / 0.76)

| dephasing | cap | mirror | gens (of 6) | runs with all 6 gens | final n | births bin / ter | Cf mean / min | purity | mirror gain | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 48 | on | 2.35 | 0% | 48.0 | 27.5 / 14.5 | 0.817 / 0.710 | 0.9999 | 0.0331 | 4.4 |
| 0.05 | 48 | off | 2.45 | 0% | 48.0 | 28.1 / 13.8 | 0.769 / 0.657 | 0.8754 | 0.0000 | 1.7 |
| 0.05 | 96 | on | 2.88 | 0% | 96.0 | 60.3 / 29.7 | 0.799 / 0.669 | 0.9999 | 0.0343 | 11.3 |
| 0.05 | 96 | off | 2.85 | 0% | 96.0 | 59.6 / 30.4 | 0.737 / 0.620 | 0.8577 | 0.0000 | 5.6 |
| 0.05 | 192 | on | 3.15 | 0% | 192.0 | 132.8 / 53.2 | 0.779 / 0.641 | 0.9999 | 0.0329 | 22.4 |
| 0.05 | 192 | off | 3.02 | 0% | 192.0 | 138.2 / 47.8 | 0.710 / 0.597 | 0.8443 | 0.0000 | 10.8 |
| 0.2 | 48 | on | 2.33 | 0% | 48.0 | 27.4 / 14.6 | 0.819 / 0.712 | 0.9999 | 0.0332 | 4.7 |
| 0.2 | 48 | off | 2.27 | 0% | 48.0 | 27.4 / 14.6 | 0.690 / 0.593 | 0.6831 | 0.0000 | 1.3 |
| 0.2 | 96 | on | 2.90 | 0% | 96.0 | 60.3 / 29.7 | 0.797 / 0.666 | 0.9999 | 0.0345 | 11.4 |
| 0.2 | 96 | off | 2.95 | 0% | 96.0 | 60.3 / 29.7 | 0.614 / 0.514 | 0.6119 | 0.0000 | 6.2 |
| 0.2 | 192 | on | 3.12 | 0% | 192.0 | 129.4 / 56.6 | 0.778 / 0.640 | 0.9999 | 0.0332 | 21.6 |
| 0.2 | 192 | off | 3.10 | 0% | 192.0 | 136.8 / 49.2 | 0.596 / 0.493 | 0.5943 | 0.0000 | 13.1 |
| 0.5 | 48 | on | 2.33 | 0% | 48.0 | 27.4 / 14.6 | 0.818 / 0.712 | 0.9999 | 0.0333 | 4.0 |
| 0.5 | 48 | off | 2.27 | 0% | 48.0 | 27.3 / 14.7 | 0.519 / 0.450 | 0.4147 | 0.0000 | 1.6 |
| 0.5 | 96 | on | 2.92 | 0% | 96.0 | 56.8 / 33.2 | 0.798 / 0.663 | 0.9999 | 0.0339 | 11.8 |
| 0.5 | 96 | off | 2.73 | 0% | 96.0 | 53.6 / 36.4 | 0.465 / 0.390 | 0.3506 | 0.0000 | 5.5 |
| 0.5 | 192 | on | 3.05 | 0% | 192.0 | 134.2 / 51.8 | 0.766 / 0.648 | 0.9999 | 0.0334 | 24.0 |
| 0.5 | 192 | off | 3.02 | 0% | 192.0 | 140.7 / 45.3 | 0.427 / 0.349 | 0.3079 | 0.0000 | 11.6 |
| 1 | 48 | on | 2.33 | 0% | 48.0 | 27.4 / 14.6 | 0.819 / 0.712 | 0.9997 | 0.0343 | 4.0 |
| 1 | 48 | off | 2.15 | 0% | 48.0 | 30.3 / 11.7 | 0.349 / 0.303 | 0.2072 | 0.0000 | 1.2 |
| 1 | 96 | on | 2.88 | 0% | 96.0 | 59.1 / 30.9 | 0.791 / 0.667 | 0.9997 | 0.0345 | 11.6 |
| 1 | 96 | off | 2.73 | 0% | 96.0 | 49.4 / 40.6 | 0.272 / 0.229 | 0.1458 | 0.0000 | 6.9 |
| 1 | 192 | on | 3.00 | 0% | 192.0 | 130.8 / 55.1 | 0.763 / 0.649 | 0.9998 | 0.0334 | 19.4 |
| 1 | 192 | off | 3.00 | 0% | 192.0 | 130.8 / 55.2 | 0.238 / 0.194 | 0.1160 | 0.0000 | 17.9 |
| 2 | 48 | on | 2.40 | 0% | 48.0 | 27.9 / 14.1 | 0.822 / 0.710 | 0.9989 | 0.0362 | 4.6 |
| 2 | 48 | off | 2.08 | 0% | 48.0 | 27.4 / 14.6 | 0.162 / 0.139 | 0.0692 | 0.0000 | 0.9 |
| 2 | 96 | on | 2.80 | 0% | 96.0 | 57.7 / 32.3 | 0.796 / 0.674 | 0.9989 | 0.0363 | 10.9 |
| 2 | 96 | off | 2.58 | 0% | 96.0 | 47.7 / 42.3 | 0.114 / 0.096 | 0.0534 | 0.0000 | 5.8 |
| 2 | 192 | on | 3.00 | 0% | 192.0 | 132.2 / 53.8 | 0.774 / 0.657 | 0.9991 | 0.0351 | 20.2 |
| 2 | 192 | off | 3.00 | 0% | 192.0 | 128.3 / 57.7 | 0.076 / 0.062 | 0.0401 | 0.0000 | 17.9 |

Reading: **all 30 settings hit the cap in 100% of runs** (the cap-hit fraction is in the CSV),
after 2.1–3.2 generations. Dephasing and cap do not control growth. The cause is the
resonance gate: founders start at a mean pairwise fidelity of 0.93, and after one generation
97–100% of pairs clear `thresh_2 = 0.70` (SYNTHETIC diagnostic, 20 seeds). Births therefore
grow with the number of pairs, which is quadratic in n, and a larger cap only adds about one
generation. The mirror effect, in contrast, grows with dephasing. With the mirror on, purity
stays at 0.998 or above at every rate. With it off, purity drops from 0.88 to 0.07 and mean Cf
from 0.77 to 0.16 (cap 48, γ 0.05 to 2.0). Without the mirror, dephasing drives membranes
toward the maximally mixed state, where they all look alike: fidelity rises, so births do
**not** slow down.

### Grid 2 (follow-up): resonance threshold × dephasing, cap 96

Since grid 1 showed the threshold is the real lever, this follow-up raises `thresh_2`
(`thresh_3 = min(0.92, thresh_2 + 0.06)`, within the Calibrator's clip bounds).

| dephasing | cap | thresh_2/3 | mirror | gens (of 6) | runs with all 6 gens | final n | births bin / ter | Cf mean / min | purity | mirror gain | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 96 | 0.80/0.86 | on | 3.85 | 10% | 93.4 | 61.3 / 26.1 | 0.818 / 0.662 | 1.0000 | 0.0208 | 14.5 |
| 0.05 | 96 | 0.80/0.86 | off | 3.88 | 8% | 92.5 | 63.9 / 22.6 | 0.758 / 0.612 | 0.8312 | 0.0000 | 6.7 |
| 0.5 | 96 | 0.80/0.86 | on | 3.70 | 5% | 96.0 | 63.9 / 26.1 | 0.797 / 0.643 | 1.0000 | 0.0209 | 14.4 |
| 0.5 | 96 | 0.80/0.86 | off | 3.15 | 0% | 96.0 | 64.9 / 25.1 | 0.417 / 0.349 | 0.3041 | 0.0000 | 5.6 |
| 2 | 96 | 0.80/0.86 | on | 3.65 | 2% | 96.0 | 63.5 / 26.5 | 0.836 / 0.680 | 0.9988 | 0.0237 | 14.0 |
| 2 | 96 | 0.80/0.86 | off | 2.58 | 0% | 96.0 | 47.7 / 42.3 | 0.114 / 0.096 | 0.0534 | 0.0000 | 5.5 |
| 0.05 | 96 | 0.85/0.91 | on | 4.42 | 22% | 86.0 | 57.6 / 22.4 | 0.831 / 0.681 | 1.0000 | 0.0191 | 13.1 |
| 0.05 | 96 | 0.85/0.91 | off | 4.58 | 22% | 88.4 | 59.3 / 23.1 | 0.745 / 0.599 | 0.8126 | 0.0000 | 5.8 |
| 0.5 | 96 | 0.85/0.91 | on | 4.65 | 30% | 88.7 | 63.2 / 19.5 | 0.820 / 0.656 | 1.0000 | 0.0195 | 14.2 |
| 0.5 | 96 | 0.85/0.91 | off | 3.67 | 8% | 96.0 | 65.6 / 24.4 | 0.407 / 0.323 | 0.2635 | 0.0000 | 4.5 |
| 2 | 96 | 0.85/0.91 | on | 4.67 | 28% | 87.7 | 59.8 / 22.0 | 0.832 / 0.689 | 0.9988 | 0.0217 | 15.3 |
| 2 | 96 | 0.85/0.91 | off | 2.73 | 0% | 96.0 | 51.4 / 38.6 | 0.101 / 0.085 | 0.0488 | 0.0000 | 6.0 |
| 0.05 | 96 | 0.88/0.92 | on | 5.22 | 57% | 64.3 | 42.2 / 16.1 | 0.852 / 0.695 | 1.0000 | 0.0170 | 10.9 |
| 0.05 | 96 | 0.88/0.92 | off | 5.12 | 48% | 70.0 | 46.3 / 17.7 | 0.763 / 0.628 | 0.7992 | 0.0000 | 4.3 |
| 0.5 | 96 | 0.88/0.92 | on | 5.30 | 62% | 64.6 | 42.2 / 16.4 | 0.846 / 0.687 | 1.0000 | 0.0177 | 11.6 |
| 0.5 | 96 | 0.88/0.92 | off | 3.75 | 5% | 96.0 | 60.5 / 29.6 | 0.401 / 0.324 | 0.2573 | 0.0000 | 4.8 |
| 2 | 96 | 0.88/0.92 | on | 5.28 | 60% | 60.4 | 39.5 / 14.9 | 0.831 / 0.684 | 0.9987 | 0.0198 | 9.8 |
| 2 | 96 | 0.88/0.92 | off | 2.85 | 0% | 96.0 | 58.5 / 31.4 | 0.089 / 0.075 | 0.0449 | 0.0000 | 4.9 |
| 0.05 | 96 | 0.90/0.92 | on | 5.50 | 72% | 54.4 | 33.4 / 15.1 | 0.835 / 0.697 | 1.0000 | 0.0164 | 9.7 |
| 0.05 | 96 | 0.90/0.92 | off | 5.25 | 60% | 64.0 | 39.4 / 18.6 | 0.740 / 0.616 | 0.7992 | 0.0000 | 3.2 |
| 0.5 | 96 | 0.90/0.92 | on | 5.60 | 78% | 41.8 | 24.5 / 11.3 | 0.848 / 0.713 | 1.0000 | 0.0167 | 8.0 |
| 0.5 | 96 | 0.90/0.92 | off | 4.40 | 22% | 90.7 | 52.6 / 32.1 | 0.362 / 0.291 | 0.2120 | 0.0000 | 5.0 |
| 2 | 96 | 0.90/0.92 | on | 5.72 | 85% | 39.5 | 24.5 / 9.0 | 0.845 / 0.701 | 0.9983 | 0.0196 | 8.5 |
| 2 | 96 | 0.90/0.92 | off | 2.98 | 0% | 96.0 | 59.0 / 30.9 | 0.081 / 0.067 | 0.0422 | 0.0000 | 5.3 |

Reading: from `thresh_2 ≥ 0.88`, populations stop running into the cap when the mirror is on.
With **thresh 0.90 / 0.92, dephasing 0.5, cap 96, mirror on**:
- 5.6 generations on average, and 78% of runs complete all 6.
- The final n is 41.8, well under the cap.
- Cf is 0.848 / 0.713 and purity 1.0000.

The same setting with the **mirror off** reaches the cap in most runs (4.4 generations, 22% of runs complete all 6, final n 90.7), with Cf 0.362 / 0.291 and purity 0.212. At dephasing 2.0 the contrast is sharper: 85% vs 0% of runs complete all 6 generations, and purity is 0.998 vs 0.042. In this regime the mirror changes both the states and the population dynamics, because mixed membranes resemble each other and pass the gate more easily. At dephasing 0.05, the mirror's effect on purity is small (1.000 vs 0.80).

**Defaults are unchanged** (γ = 0.05, cap 48, thresholds 0.70 / 0.76). They are the original prototype's
parameters and keep the v1/v2 comparison above reproducible. The regime above is a
recommended **experiment setting**, not a better model. All of these numbers are SYNTHETIC
and come from one seed.

## Known limitations

- **Header floor `sum|psi_i|^2 + S(R) = 1`: UNRUN, not implemented.** The source does not define R or S. The only definitions that make the identity hold for every state are tautologies, such as purity + (1 − purity) = 1, and a tautology carries no information. Its first term, Σ ρ_ii = tr ρ = 1, is just normalization. v2 therefore reports what can be defined rigorously, as separate SYNTHETIC diagnostics: tr ρ (the "Max \|tr ρ − 1\|" row), purity tr ρ² and normalized von Neumann entropy S/ln N.
- In v2 at default settings, the mirrors (per-step emp_floor mixing, per-generation eureka step, post-birth) outweigh the weak dephasing (γ = 0.05), so states stay nearly pure (purity 0.9999). The mirror's effect shows most clearly in the ablation.
- At default settings the cap (48) is still reached by generation 2 or 3, so most of the 6 generations are never run (the same happens in v1). The parameter sweep shows why (nearly all pairs pass `thresh_2 = 0.70`) and which thresholds avoid it.
- Cf is phase-blind on pure states: any uniform-magnitude state has Cf = 1.
- Parameters and thresholds are the original prototype's heuristics. None of them is fitted to data, and no metric here corresponds to a physical, biological or cognitive quantity.

## Evidence tags

See [sparkainlp-x/.github: Evidence tags](https://github.com/sparkainlp-x/.github#evidence-tags).

| Tag | Meaning here |
|---|---|
| **SYNTHETIC** | Produced by a real run of this code (every number above) |
| **TARGET** | A design goal; not yet achieved or measured (none claimed) |
| **UNRUN** | Stated but not implemented or run (the header "floor") |

## License and citation

MIT © 2026 Jean-François Brisson / Spark AI NLP. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

Archived on Zenodo: concept DOI [10.5281/zenodo.22999270](https://doi.org/10.5281/zenodo.22999270) (all versions); v0.2.0: [10.5281/zenodo.22999271](https://doi.org/10.5281/zenodo.22999271). Changes are listed in [CHANGELOG.md](CHANGELOG.md).
