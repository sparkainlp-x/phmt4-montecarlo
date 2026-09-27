# phmt4-montecarlo

PHMT-4 Monte Carlo: **a heuristic, classical numpy simulation**. It has 32-dimensional
"membranes", dephasing evolution, an eigenvalue-boost "crystallization mirror",
binary/ternary births gated by resonance thresholds, an Observer that reports the coherent
fraction Cf, and a Calibrator that adapts parameters.

This branch holds **v2** (`phmt4/v2.py`), a density-matrix redesign that fixes the design
flaws found in the founder's original. The faithful original with minimal fixes (v1) and its
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
python -m pytest              # 14 tests, a few seconds
python scripts/v2_checks.py   # integrator / mirror checks
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
- **Output:** a seeded CLI with `--csv` (per-sample rows) and `--json` (means, runtime, config), both tagged SYNTHETIC.

### Integrator check (SYNTHETIC, [results/v2_checks.txt](results/v2_checks.txt))

| dt | max error vs dt = T/3200 | ratio |
|---|---|---|
| T/20 (default, dt = 0.02) | 4.43e-07 | – |
| T/40 | 1.11e-07 | 4.01 |
| T/80 | 2.76e-08 | 4.00 |

A ratio of 4 means second-order convergence. Without dephasing, the overlap with H's dominant
eigenvector stays at 0.4114 after 60 exact steps (v1's Euler step drives it to 1.0000).

## Known limitations

- **Header floor `sum|psi_i|^2 + S(R) = 1`: UNRUN, not implemented.** The source does not define R or S. The only definitions that make the identity hold for every state are tautologies, such as purity + (1 − purity) = 1, and a tautology carries no information. Its first term, Σ ρ_ii = tr ρ = 1, is just normalization. v2 therefore reports what can be defined rigorously, as separate SYNTHETIC diagnostics: tr ρ (the "Max \|tr ρ − 1\|" row), purity tr ρ² and normalized von Neumann entropy S/ln N.
- In v2 at default settings, the mirrors (per-step emp_floor mixing, per-generation eureka step, post-birth) outweigh the weak dephasing (γ = 0.05), so states stay nearly pure (purity 0.9999). The mirror's effect shows most clearly in the ablation.
- The cap (48) is still reached by generation 2 or 3, so most of the 6 generations are never run (the same happens in v1).
- Cf is phase-blind on pure states: any uniform-magnitude state has Cf = 1.
- Parameters and thresholds are the founder's heuristics. None of them is fitted to data, and no metric here corresponds to a physical, biological or cognitive quantity.

## Evidence tags

See [sparkainlp-x/.github: Evidence tags](https://github.com/sparkainlp-x/.github#evidence-tags).

| Tag | Meaning here |
|---|---|
| **SYNTHETIC** | Produced by a real run of this code (every number above) |
| **TARGET** | A design goal; not yet achieved or measured (none claimed) |
| **UNRUN** | Stated but not implemented or run (the header "floor") |

## License and citation

MIT © 2026 Jean-François Brisson / Spark AI NLP. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
