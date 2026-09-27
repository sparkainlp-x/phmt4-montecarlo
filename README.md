# phmt4-montecarlo

PHMT-4 Monte Carlo: **a heuristic, classical numpy simulation**. It has 32-dimensional
complex state vectors ("membranes"), a quantum-jump-style dephasing evolution, an
eigenvalue-boost "crystallization mirror", binary/ternary births gated by resonance
thresholds, an Observer that reports the coherent fraction Cf, and a Calibrator that adapts
parameters.

This tag (`v1-original`) is the founder's original script with **minimal, documented fixes**
(`phmt4/v1.py`). The verbatim original is in [`original/`](original/phmt4_montecarlo_original.py).

## What it is NOT

- **Not a consciousness theory.** The source header says so, and nothing here measures or models consciousness.
- **Not quantum hardware and not a quantum computer.** It is ordinary floating-point linear algebra on a CPU. Words like "jump", "dephasing" or "density matrix" name the math it borrows, not a physical device.
- **Not biology or medicine.** "Membrane", "birth" and "fecundity" are metaphors for vectors and population rules.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python phmt4_montecarlo.py --samples 40 --seed 42 --csv out.csv --json out.json
python -m pytest            # 9 tests, a few seconds
python scripts/v1_checks.py # numerical checks behind "Known limitations"
```

## Results: default run (40 samples, seed 42), all **SYNTHETIC**

Machine: 8-core Linux box, Python 3.13.5, numpy 2.5.3, pandas 3.0.6. The full output is in
[`results/v1_seed42.txt`](results/v1_seed42.txt), with per-sample rows in [`results/v1_seed42.csv`](results/v1_seed42.csv).

| Metric (mean over 40 runs) | Value | Tag |
|---|---|---|
| Final population | 48.00 (= cap in every run) | SYNTHETIC |
| Binary births | 28.25 | SYNTHETIC |
| Ternary births | 13.75 | SYNTHETIC |
| Jumps | 0.57 | SYNTHETIC |
| Final Cf (mean / std / min) | 0.971 / 0.025 / 0.808 | SYNTHETIC |
| Founder / binary / ternary Cf | 0.960 / 0.971 / 0.970 | SYNTHETIC |
| Protection held | 100.0% | SYNTHETIC |
| Generations completed (of 6) | 2.1 | SYNTHETIC |
| Runtime, original script | 10.44 s | SYNTHETIC |
| Runtime, v1 with caching (same output) | 6.69 s | SYNTHETIC |

The v1 output is **bit-identical** to the original script at seed 42 (checked by
`test_v1_matches_original_bit_for_bit`).

## v1 fixes (minimal)

| # | Issue | Fix |
|---|---|---|
| F1 | A module-level `rng = default_rng(42)` was shared across calls, `C.seed` was unused, and `run_multigen()` was unseeded by default | `monte_carlo(n, seed=C.seed)` and `run_multigen(rng=None, seed=None)` build their own Generator. Output at seed 42 is unchanged. |
| F2 | `evolve()` rebuilt H and the 32 dephasing operators on every call, `jump_step()` rebuilt the effective Hamiltonian on every step, `crystallize()` computed `to_rho` twice, and ternary resonances were recomputed per triple | Cached H, the operators and H_eff per level, computed `to_rho` once, and built a pairwise resonance matrix once per generation. The arithmetic is the same, so results are bit-identical. Runtime went from 10.44 s to 6.69 s (SYNTHETIC). |
| F3 | `__main__` hard-coded 40 samples | Added a CLI: `--samples`, `--seed`, `--csv`, `--json`, `--quiet`. |

Verified and **not changed**:
- The ternary `break` on cap exits the single `combinations(..., 3)` loop, which is the whole ternary pass, so the cap is enforced correctly. The binary loop works the same way. Population never exceeds the cap (tested).
- `purity()` and `self_knowing()` are unused. They are kept and marked as unused.

## Known limitations (verified numerically, SYNTHETIC; see `results/v1_checks.txt`)

1. **The crystallization mirror is a no-op on pure states.** ρ = |ψ⟩⟨ψ| has rank 1, so boosting its eigenvalues keeps the same top eigenvector. Across 1000 random states and 5 strengths, `resonance(psi, crystallize(psi))` fell between 0.9999999999999993 and 1.0000000000000009. The only change is an arbitrary global phase from `eigh`. Ablation: replacing `crystallize` with the identity gives **identical** births (28.25 / 13.75) and jumps (0.575) at seed 42, and mean Cf moves only from 0.970738 to 0.970665 (the phase leaks into births through superposition). The Calibrator action "strengthen crystallization mirror" therefore has no measurable effect on the states.
2. **The no-jump step is first-order and non-unitary.** `psi - i dt H_eff psi` followed by renormalization multiplies each H-eigencomponent by about sqrt(1 + E^2 dt^2). With E up to about 31 and dt = 0.02, that acts as **power iteration**: a founder's overlap with H's dominant eigenvector goes from 0.44 to 0.994 in 20 steps (one generation) and to 1.0000 by step 40. The exact unitary keeps it at 0.44. So v1's Cf of about 0.975 is the Cf of H's dominant eigenvector (0.9753), which is an **integrator artifact**. This is also why nearly all pairs pass the resonance thresholds and the cap is reached by generation 2.
3. **Jumps collapse to a basis state.** L_i = sqrt(γ)|i⟩⟨i| projects ψ onto e_i (Cf = 0). Every run with a low min Cf had at least one jump.
4. **The floor `sum|psi_i|^2 + S(R) = 1` in the header is not computed** anywhere in the code: **UNRUN / not implemented**.
5. **Cf is phase-blind on pure states.** Cf = ((Σ|ψ_i|)² − 1)/(N − 1), so any uniform-magnitude state has Cf = 1.
6. **Cross-platform reproducibility:** the global phase returned by LAPACK `eigh` can differ between BLAS builds, and it feeds into births. Results are deterministic on a given platform (tested), but exact values may differ elsewhere.

## Evidence tags

See [sparkainlp-x/.github: Evidence tags](https://github.com/sparkainlp-x/.github#evidence-tags).

| Tag | Meaning here |
|---|---|
| **SYNTHETIC** | Produced by a real run of this code (numbers above) |
| **TARGET** | A design goal; not yet achieved or measured (none claimed) |
| **UNRUN** | Stated or scripted but not run/implemented (the header "floor") |

## License and citation

MIT © 2026 Jean-François Brisson / Spark AI NLP. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
