# ============================================================
# PHMT-4 Monte Carlo  (v1: faithful original + minimal fixes)
# Protected membranes + binary/ternary fecundity
# Crystallization mirror + Observer + Calibrator
# ============================================================
# Heuristic dynamical architecture. Not a consciousness theory.
# Floor:  sum |psi_i|^2 + S(R) = 1
#   -> UNRUN: stated in the original header, not computed anywhere.
# ============================================================
# Changes vs original/phmt4_montecarlo_original.py (see README "v1 fixes"):
#   F1 seed: no shared module-level rng; monte_carlo(seed=C.seed) and
#      run_multigen(seed=...) build their own Generator. Default output with
#      seed 42 is bit-identical to the original script.
#   F2 perf: H(level), the dephasing ops and the effective Hamiltonian are
#      cached; to_rho is computed once in crystallize; pairwise resonances are
#      computed once per generation. Same arithmetic, identical results.
#   F3 CLI: --samples / --seed / --csv / --json; no hard-coded 40.
# The dynamics are otherwise unchanged (see README "Known limitations").

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd


# -------------------- Config --------------------
class C:
    N = 32
    dt = 0.02
    steps_per_gen = 20
    dephasing = 0.05
    eureka_str = 0.78
    emp_floor = 0.05
    n_initial = 6
    n_generations = 6
    max_membranes = 48
    n_samples = 40
    seed = 42


N = C.N

BASE_PARAMS = {
    "thresh_2": 0.70,
    "thresh_3": 0.76,
    "fecund_p2": 0.28,
    "fecund_p3": 0.18,
    "mix_2": 0.14,
    "mix_3": 0.16,
    "post_birth_concentrate": 0.58,
    "max_membranes": C.max_membranes,
}


# -------------------- Spectral / state helpers --------------------
def normalize(psi):
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-12 else np.ones(N, dtype=complex) / np.sqrt(N)


def to_rho(psi):
    psi = normalize(psi)
    return np.outer(psi, psi.conj())


def coherent_fraction(rho):
    """l1-norm of coherence divided by its maximum (N-1). In [0, 1]."""
    rho = 0.5 * (rho + rho.conj().T)
    off = np.abs(rho - np.diag(np.diag(rho).real))
    return min(1.0, float(np.sum(off)) / (N - 1))


def purity(rho):
    """Unused by the simulation (kept from the original)."""
    rho = 0.5 * (rho + rho.conj().T)
    return float(np.real(np.trace(rho @ rho)))


def self_knowing(rho):
    """Unused by the simulation (kept from the original)."""
    ev = np.clip(np.real(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))), 1e-12, None)
    ev /= ev.sum()
    S = -np.sum(ev * np.log(ev + 1e-12))
    return max(0.0, 1.5 - 0.8 * S)


def resonance(a, b):
    return float(np.abs(np.vdot(normalize(a), normalize(b))) ** 2)


def geometric_mean_3(a, b, c):
    return (resonance(a, b) * resonance(b, c) * resonance(c, a)) ** (1.0 / 3.0)


def crystallize(psi, strength=0.58):
    """Eigenvalue-boost crystallization mirror.

    Known limitation: rho = |psi><psi| is rank 1, so the boosted matrix has the
    same top eigenvector and this returns psi up to a global phase
    (resonance(psi, crystallize(psi)) == 1 to ~1e-15). See README.
    """
    if strength <= 0:
        return normalize(psi)
    r = to_rho(psi)  # F2: computed once (original called to_rho twice)
    rho = 0.5 * (r + r.conj().T)
    ev, vec = np.linalg.eigh(rho)
    gamma = 1.0 + 2.0 * float(strength)
    boosted = np.clip(ev, 1e-12, None) ** gamma
    boosted /= boosted.sum()
    rho_c = vec @ np.diag(boosted) @ vec.conj().T
    ev2, vec2 = np.linalg.eigh(0.5 * (rho_c + rho_c.conj().T))
    return normalize(vec2[:, -1])


def concentrate(psi, strength=0.78):
    return crystallize(psi, strength=strength)


def build_H(level=3):
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = 0.28 * np.sin(2 * np.pi * i / N) + 0.05 * level
    for i in range(N):
        for j in range(i + 1, N):
            s = 1.8 * np.exp(-0.09 * abs(i - j)) * (1 + 0.07 * level)
            H[i, j] = H[j, i] = s
    return H


def dephasing_ops(rate):
    ops = []
    for i in range(N):
        L = np.zeros((N, N), dtype=complex)
        L[i, i] = np.sqrt(rate)
        ops.append(L)
    return ops


@lru_cache(maxsize=None)
def _cached_ops(rate):
    return tuple(dephasing_ops(rate))


@lru_cache(maxsize=None)
def _cached_H(level):
    H = build_H(level)
    H.setflags(write=False)
    return H


def effective_H(H, ops):
    He = H.copy().astype(complex)
    for L in ops:
        He -= 0.5j * (L.conj().T @ L)
    return He


@lru_cache(maxsize=None)
def _cached_He(level, rate):
    He = effective_H(_cached_H(level), _cached_ops(rate))
    He.setflags(write=False)
    return He


def jump_step(psi, H, ops, dt, rng, He=None):
    """One quantum-jump-style step.

    Heuristic: the no-jump branch is a first-order (Euler) step with the
    non-Hermitian effective Hamiltonian, followed by renormalization. It is
    neither unitary nor an exact trajectory integrator.
    """
    psi = normalize(psi)
    if He is None:
        He = effective_H(H, ops)
    probs = np.array([max(0.0, dt * np.real(np.vdot(L @ psi, L @ psi))) for L in ops])
    total = probs.sum()
    if total > 0 and rng.random() < total:
        probs /= total
        idx = rng.choice(len(ops), p=probs)
        return normalize(ops[idx] @ psi), True
    return normalize(psi - 1j * dt * (He @ psi)), False


def evolve(psi, level, steps, rng):
    ops = _cached_ops(C.dephasing)
    H = _cached_H(level)
    He = _cached_He(level, C.dephasing)
    jumps = 0
    for _ in range(steps):
        psi, jumped = jump_step(psi, H, ops, C.dt, rng, He=He)
        if jumped:
            jumps += 1
            psi = crystallize(psi, 0.35)   # jump mirror
        psi = normalize((1 - C.emp_floor) * psi + C.emp_floor * concentrate(psi, 0.40))
    return normalize(psi), jumps


# -------------------- Birth with crystallization mirror --------------------
def birth_binary(a, b, mix, mirror, gen, kid_id):
    raw = normalize((1 - mix) * normalize(a["psi"]) + mix * normalize(b["psi"]))
    return {
        "psi": crystallize(raw, mirror),
        "level": max(a.get("level", 1), b.get("level", 1)) + 1,
        "generation": gen,
        "protection": True,
        "id": kid_id,
        "type": "binary",
        "crystallized": True,
    }


def birth_ternary(a, b, c, mix, mirror, gen, kid_id):
    raw = normalize(
        (1 - mix) * normalize(a["psi"])
        + (mix / 2) * normalize(b["psi"])
        + (mix / 2) * normalize(c["psi"])
    )
    return {
        "psi": crystallize(raw, mirror),
        "level": max(a.get("level", 1), b.get("level", 1), c.get("level", 1)) + 1,
        "generation": gen,
        "protection": True,
        "id": kid_id,
        "type": "ternary",
        "crystallized": True,
    }


# -------------------- Observer --------------------
class ObserverAgent:
    def __init__(self):
        self.last = None

    def observe(self, population, gen=None):
        if not population:
            self.last = {"n": 0, "mean_cf": 0, "std_cf": 0, "min_cf": 0, "by_type": {}}
            return self.last
        cfs = np.array([coherent_fraction(to_rho(m["psi"])) for m in population])
        by_type = defaultdict(list)
        for m, cf in zip(population, cfs):
            t = m.get("type") or ("founder" if m.get("generation", 0) == 0 else "binary")
            by_type[t].append(cf)
        typed = {}
        for t in ("founder", "binary", "ternary"):
            vals = np.array(by_type.get(t, []), dtype=float)
            if len(vals) == 0:
                typed[t] = {"n": 0, "mean_cf": 0.0, "std_cf": 0.0, "min_cf": 0.0}
            else:
                typed[t] = {
                    "n": int(len(vals)),
                    "mean_cf": float(vals.mean()),
                    "std_cf": float(vals.std()),
                    "min_cf": float(vals.min()),
                }
        n_prot = sum(1 for m in population if m.get("protection", False))
        report = {
            "generation_index": gen,
            "n": len(population),
            "mean_cf": float(cfs.mean()),
            "std_cf": float(cfs.std()),
            "min_cf": float(cfs.min()),
            "max_cf": float(cfs.max()),
            "protection_fraction": n_prot / len(population),
            "by_type": typed,
        }
        self.last = report
        return report


# -------------------- Calibrator --------------------
class CalibratorAgent:
    CLIP = {
        "thresh_2": (0.55, 0.90),
        "thresh_3": (0.65, 0.92),
        "fecund_p2": (0.05, 0.48),
        "fecund_p3": (0.03, 0.35),
        "post_birth_concentrate": (0.30, 0.90),
    }

    def __init__(self, base):
        self.params = dict(base)
        self.history = []

    def act(self, report):
        p = dict(self.params)
        acts = []
        if not report or report.get("n", 0) == 0:
            return p
        n = report["n"]
        fill = n / p["max_membranes"]
        min_cf = report["min_cf"]
        std_cf = report["std_cf"]
        mean_cf = report["mean_cf"]

        if fill > 0.75:
            old = p["fecund_p2"]
            p["fecund_p2"] = max(0.08, old * 0.72)
            acts.append("slow binary growth")
        if fill > 0.90:
            p["fecund_p3"] = max(0.04, p["fecund_p3"] * 0.60)
            acts.append("slow ternary growth")
        if min_cf < 0.86 or std_cf > 0.06:
            old = p["post_birth_concentrate"]
            p["post_birth_concentrate"] = min(0.86, old + 0.07)
            acts.append("strengthen crystallization mirror")
        tern = report.get("by_type", {}).get("ternary", {})
        if tern.get("n", 0) > 0 and tern.get("min_cf", 1.0) < 0.84:
            p["thresh_3"] = min(0.88, p["thresh_3"] + 0.02)
            p["fecund_p3"] = max(0.04, p["fecund_p3"] * 0.75)
            acts.append("raise ternary bar")
        if fill < 0.40 and mean_cf > 0.96 and std_cf < 0.04:
            p["fecund_p2"] = min(0.42, p["fecund_p2"] * 1.08)
            acts.append("encourage binary")

        for k, (lo, hi) in self.CLIP.items():
            p[k] = float(np.clip(p[k], lo, hi))
        self.params = p
        self.history.append({"gen": report.get("generation_index"), "n": n, "min_cf": min_cf, "acts": acts})
        return p


# -------------------- One multi-generation run --------------------
def run_multigen(rng=None, seed=None):
    """One run. Pass a Generator (``rng``) or an int ``seed``; with neither,
    the run is unseeded (non-reproducible), as in the original."""
    if rng is None:
        rng = np.random.default_rng(seed)
    observer = ObserverAgent()
    calibrator = CalibratorAgent(BASE_PARAMS)
    params = dict(BASE_PARAMS)

    population = []
    for i in range(C.n_initial):
        phase = 0.10 * np.arange(N) + 0.22 * rng.normal(size=N)
        population.append({
            "psi": normalize(np.exp(1j * phase)),
            "level": int(rng.integers(2, 5)),
            "generation": 0,
            "protection": True,
            "id": f"G0_M{i}",
            "type": "founder",
        })

    births_2 = births_3 = jumps_tot = 0
    history = []
    max_pop_seen = len(population)

    for gen in range(C.n_generations):
        if len(population) >= params["max_membranes"]:
            break
        for m in population:
            m["psi"], j = evolve(m["psi"], m["level"], C.steps_per_gen, rng)
            jumps_tot += j
            m["psi"] = concentrate(m["psi"], C.eureka_str)

        # F2: pairwise resonance computed once (symmetric; bit-identical values)
        npop = len(population)
        R = np.ones((npop, npop))
        for i, j in combinations(range(npop), 2):
            R[i, j] = R[j, i] = resonance(population[i]["psi"], population[j]["psi"])

        newborns = []
        cap = params["max_membranes"]
        # binary (primary yin-yang channel)
        for i, j in combinations(range(npop), 2):
            if npop + len(newborns) >= cap:
                break
            if R[i, j] >= params["thresh_2"]:
                if rng.random() < params["fecund_p2"]:
                    newborns.append(birth_binary(
                        population[i], population[j],
                        params["mix_2"], params["post_birth_concentrate"],
                        gen + 1, f"G{gen+1}_B2_{len(newborns)}"
                    ))
                    births_2 += 1
        # ternary (secondary). The break exits the single combinations loop,
        # which is the whole ternary pass, so the cap is enforced correctly.
        if npop >= 3:
            for i, j, k in combinations(range(npop), 3):
                if npop + len(newborns) >= cap:
                    break
                r3 = (R[i, j] * R[j, k] * R[k, i]) ** (1.0 / 3.0)
                if r3 >= params["thresh_3"] and rng.random() < params["fecund_p3"]:
                    newborns.append(birth_ternary(
                        population[i], population[j], population[k],
                        params["mix_3"], params["post_birth_concentrate"],
                        gen + 1, f"G{gen+1}_B3_{len(newborns)}"
                    ))
                    births_3 += 1

        population.extend(newborns)
        max_pop_seen = max(max_pop_seen, len(population))
        report = observer.observe(population, gen)
        params = calibrator.act(report)
        history.append({
            "generation": gen,
            "n": report["n"],
            "mean_cf": report["mean_cf"],
            "std_cf": report["std_cf"],
            "min_cf": report["min_cf"],
            "births": len(newborns),
            "mirror": params["post_birth_concentrate"],
        })

    final = observer.observe(population, history[-1]["generation"] if history else 0)
    return {
        "final_n": final["n"],
        "births_2": births_2,
        "births_3": births_3,
        "jumps": jumps_tot,
        "mean_cf": final["mean_cf"],
        "std_cf": final["std_cf"],
        "min_cf": final["min_cf"],
        "prot": final["protection_fraction"],
        "gens": len(history),
        "bin_mean": final["by_type"]["binary"]["mean_cf"],
        "bin_std": final["by_type"]["binary"]["std_cf"],
        "ter_mean": final["by_type"]["ternary"]["mean_cf"],
        "ter_std": final["by_type"]["ternary"]["std_cf"],
        "found_mean": final["by_type"]["founder"]["mean_cf"],
        "max_pop": max_pop_seen,
        "history": history,
    }


# -------------------- Monte Carlo --------------------
def monte_carlo(n_samples=C.n_samples, seed=C.seed, verbose=True):
    """Run ``n_samples`` independent runs from one Generator seeded with ``seed``.

    With seed=42 this reproduces the original script's output exactly."""
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)
    rows = []
    log("=" * 68)
    log("PHMT-4  |  Protection + Fecundity + Crystallization Mirror")
    log("Observer + Calibrator  |  binary primary, ternary secondary")
    log("=" * 68)
    log(f"Samples {n_samples} | gens {C.n_generations} | start {C.n_initial} | cap {C.max_membranes} | seed {seed}")
    log("-" * 68)
    for i in range(n_samples):
        res = run_multigen(rng)
        flat = {k: v for k, v in res.items() if k not in ("history", "max_pop")}
        rows.append(flat)
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{n_samples}")
    df = pd.DataFrame(rows)
    log("\nRESULTS")
    log(f"Mean final n                 : {df.final_n.mean():.2f}")
    log(f"Mean binary births           : {df.births_2.mean():.2f}")
    log(f"Mean ternary births          : {df.births_3.mean():.2f}")
    log(f"Mean jumps                   : {df.jumps.mean():.2f}")
    log(f"Mean final Cf                : {df.mean_cf.mean():.3f}")
    log(f"Mean std Cf                  : {df.std_cf.mean():.3f}")
    log(f"Mean min Cf                  : {df.min_cf.mean():.3f}")
    log(f"Mean founder Cf              : {df.found_mean.mean():.3f}")
    log(f"Mean binary-offspring Cf     : {df.bin_mean.mean():.3f}  (std {df.bin_std.mean():.3f})")
    log(f"Mean ternary-offspring Cf    : {df.ter_mean.mean():.3f}  (std {df.ter_std.mean():.3f})")
    log(f"Protection held              : {100*df.prot.mean():.1f}%")
    log(f"Mean generations completed   : {df.gens.mean():.1f}")
    log("Finished:", datetime.now().isoformat())
    log("Scope: heuristic architecture. Not a consciousness theory.")
    return df


def _write_outputs(df, args, runtime_s, version):
    if args.csv:
        df.to_csv(args.csv, index=False)
    if args.json:
        summary = {
            "version": version,
            "evidence_tag": "SYNTHETIC",
            "samples": args.samples,
            "seed": args.seed,
            "runtime_s": round(runtime_s, 3),
            "means": {k: float(v) for k, v in df.mean(numeric_only=True).items()},
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PHMT-4 Monte Carlo v1 (heuristic classical simulation)")
    ap.add_argument("--samples", type=int, default=C.n_samples)
    ap.add_argument("--seed", type=int, default=C.seed)
    ap.add_argument("--csv", help="write per-sample rows to this CSV file")
    ap.add_argument("--json", help="write a summary (means, runtime) to this JSON file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    df = monte_carlo(args.samples, seed=args.seed, verbose=not args.quiet)
    runtime = time.perf_counter() - t0
    if not args.quiet:
        print(df.describe().T[["mean", "std", "min", "max"]])
        print(f"Runtime: {runtime:.2f} s (SYNTHETIC run, this machine)")
    _write_outputs(df, args, runtime, "v1")
    return df


if __name__ == "__main__":
    main()
