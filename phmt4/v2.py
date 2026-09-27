# ============================================================
# PHMT-4 Monte Carlo  (v2: density-matrix redesign)
# v1 (faithful original + minimal fixes) is preserved at git tag v1-original.
# ============================================================
# Heuristic, classical numpy simulation. Not a consciousness theory, not
# quantum hardware, not biology.
#
# What changes vs v1-original (rationale in README "v2 design"):
#   * Each membrane is a 32x32 density matrix rho (Hermitian, PSD, trace 1),
#     not a state vector, so dephasing produces genuinely mixed states.
#   * Evolution is the Lindblad master equation with H(level) and the same
#     dephasing operators L_i = sqrt(gamma)|i><i| as v1, integrated with a
#     Strang splitting D(dt/2) . U(dt) . D(dt/2):
#       - U(dt) = exp(-i H dt) is exact (eigendecomposition of Hermitian H);
#       - D(t) multiplies off-diagonals by exp(-gamma t), the exact solution
#         of the pure-dephasing part.
#     Each factor is completely positive and trace preserving, so trace,
#     Hermiticity and positivity hold by construction (the stochastic jumps of
#     v1 are replaced by their ensemble average; there is no jump counter).
#   * Crystallization mirror: rho -> rho^g / tr(rho^g), g = 1 + 2*strength.
#     On a pure state it is the identity (nothing to sharpen); on a mixed
#     state it raises purity. Its effect is measured (purity gain).
#   * Resonance between membranes: Uhlmann fidelity, which equals
#     |<a|b>|^2 on pure states (v1's resonance).
#   * Births: convex mixture of the parents' density matrices, then the
#     post-birth mirror. A coherent superposition of mixed states is not well
#     defined, a convex mixture is.
#   * All membranes are evolved in one batched numpy pass; U(level) cached.
#   * Header "Floor: sum|psi_i|^2 + S(R) = 1" stays UNRUN (see README).
# ============================================================

import argparse
import json
import time
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd



# -------------------- shared pieces (unchanged from v1-original) --------------------
BASE_PARAMS = {
    "thresh_2": 0.70,
    "thresh_3": 0.76,
    "fecund_p2": 0.28,
    "fecund_p3": 0.18,
    "mix_2": 0.14,
    "mix_3": 0.16,
    "post_birth_concentrate": 0.58,
    "max_membranes": 48,
}


def build_H(level=3, N=32):
    """Same Hamiltonian as v1-original (vectorized; identical values)."""
    i = np.arange(N)
    H = 1.8 * np.exp(-0.09 * np.abs(i[:, None] - i[None, :])) * (1 + 0.07 * level)
    H = H.astype(complex)
    np.fill_diagonal(H, 0.28 * np.sin(2 * np.pi * i / N) + 0.05 * level)
    return H


class CalibratorAgent:
    """Same rules and clipping bounds as v1-original."""
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
        min_cf, std_cf, mean_cf = report["min_cf"], report["std_cf"], report["mean_cf"]
        if fill > 0.75:
            p["fecund_p2"] = max(0.08, p["fecund_p2"] * 0.72)
            acts.append("slow binary growth")
        if fill > 0.90:
            p["fecund_p3"] = max(0.04, p["fecund_p3"] * 0.60)
            acts.append("slow ternary growth")
        if min_cf < 0.86 or std_cf > 0.06:
            p["post_birth_concentrate"] = min(0.86, p["post_birth_concentrate"] + 0.07)
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


@dataclass(frozen=True)
class Config:
    N: int = 32
    dt: float = 0.02
    steps_per_gen: int = 20
    dephasing: float = 0.05
    eureka_str: float = 0.78
    emp_floor: float = 0.05
    emp_str: float = 0.40
    n_initial: int = 6
    n_generations: int = 6
    max_membranes: int = 48
    n_samples: int = 40
    seed: int = 42
    mirror: bool = True  # False = ablation: every crystallization is skipped
    thresh_2: float = 0.70  # binary resonance threshold (BASE_PARAMS default)
    thresh_3: float = 0.76  # ternary resonance threshold (BASE_PARAMS default)


DEFAULT = Config()


# -------------------- density-matrix helpers --------------------
def herm(rho):
    return 0.5 * (rho + np.conj(np.swapaxes(rho, -1, -2)))


def pure_rho(psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def coherent_fraction(rho):
    """l1-norm of coherence / (N-1), in [0, 1]. Works on (N,N) or (n,N,N)."""
    N = rho.shape[-1]
    a = np.abs(rho)
    off = a.sum(axis=(-1, -2)) - np.trace(a, axis1=-2, axis2=-1)
    return np.clip(off / (N - 1), 0.0, 1.0)


def purity(rho):
    return np.real(np.einsum("...ij,...ji->...", rho, rho))


def entropy_normalized(rho):
    """von Neumann entropy divided by ln N, in [0, 1]."""
    N = rho.shape[-1]
    ev = np.clip(np.linalg.eigvalsh(herm(rho)), 0.0, None)
    ev = ev / ev.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -np.where(ev > 0, ev * np.log(ev), 0.0).sum(axis=-1)
    return s / np.log(N)


def crystallize(rho, strength):
    """Eigenvalue-boost mirror rho^g / tr(rho^g), g = 1 + 2*strength.
    Accepts (N,N) or (n,N,N). Returns a valid density matrix."""
    if strength <= 0:
        return rho
    ev, vec = np.linalg.eigh(herm(rho))
    ev = np.clip(ev, 0.0, None) ** (1.0 + 2.0 * float(strength))
    ev = ev / ev.sum(axis=-1, keepdims=True)
    out = (vec * ev[..., None, :]) @ np.conj(np.swapaxes(vec, -1, -2))
    return herm(out)


def _sqrtm_psd(rho):
    ev, vec = np.linalg.eigh(herm(rho))
    ev = np.sqrt(np.clip(ev, 0.0, None))
    return (vec * ev[..., None, :]) @ np.conj(np.swapaxes(vec, -1, -2))


def fidelity_matrix(rhos):
    """Pairwise Uhlmann fidelity F(a,b) = (tr sqrt(sqrt(a) b sqrt(a)))^2."""
    n = len(rhos)
    F = np.ones((n, n))
    if n < 2:
        return F
    sq = _sqrtm_psd(rhos)
    ii, jj = np.triu_indices(n, 1)
    M = herm(sq[ii] @ rhos[jj] @ sq[ii])
    ev = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    f = np.clip(np.sqrt(ev).sum(axis=-1) ** 2, 0.0, 1.0)
    F[ii, jj] = f
    F[jj, ii] = f
    return F


def resonance(a, b):
    return float(fidelity_matrix(np.stack([a, b]))[0, 1])


# -------------------- dynamics --------------------
@lru_cache(maxsize=None)
def unitary(level, dt, N=32):
    E, V = np.linalg.eigh(build_H(level, N))
    U = (V * np.exp(-1j * E * dt)) @ V.conj().T
    U.setflags(write=False)
    return U


@lru_cache(maxsize=None)
def dephasing_mask(N, gamma, t):
    M = np.full((N, N), np.exp(-gamma * t))
    np.fill_diagonal(M, 1.0)
    M.setflags(write=False)
    return M


def evolve(rhos, levels, cfg=DEFAULT):
    """Evolve a stack (n,N,N) for one generation (steps_per_gen steps)."""
    U = np.stack([unitary(int(l), cfg.dt, cfg.N) for l in levels])
    Uh = np.conj(np.swapaxes(U, -1, -2))
    M = dephasing_mask(cfg.N, cfg.dephasing, cfg.dt / 2)
    emp = cfg.emp_str if cfg.mirror else 0.0
    for _ in range(cfg.steps_per_gen):
        rhos = M * rhos
        rhos = U @ rhos @ Uh
        rhos = M * rhos
        if emp > 0 and cfg.emp_floor > 0:
            rhos = (1 - cfg.emp_floor) * rhos + cfg.emp_floor * crystallize(rhos, emp)
        rhos = herm(rhos)
    return rhos


def _triples(n):
    """All i<j<k index triples of range(n), in itertools.combinations order."""
    ii, jj = np.triu_indices(n, 1)          # pairs in combinations order
    cnt = n - 1 - jj                        # number of k > j for each pair
    keep = cnt > 0
    ii, jj, cnt = ii[keep], jj[keep], cnt[keep]
    rep_i, rep_j = np.repeat(ii, cnt), np.repeat(jj, cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    kk = rep_j + 1 + (np.arange(cnt.sum()) - start)
    return np.stack([rep_i, rep_j, kk], axis=1)


# -------------------- run --------------------
def _observe(rhos, types, protection):
    cf = coherent_fraction(rhos)
    pur = purity(rhos)
    out = {"n": len(rhos), "mean_cf": float(cf.mean()), "std_cf": float(cf.std()),
           "min_cf": float(cf.min()), "max_cf": float(cf.max()),
           "protection_fraction": float(np.mean(protection)),
           "mean_purity": float(pur.mean()), "min_purity": float(pur.min()),
           "by_type": {}}
    types = np.asarray(types)
    for t in ("founder", "binary", "ternary"):
        v = cf[types == t]
        out["by_type"][t] = ({"n": int(len(v)), "mean_cf": float(v.mean()), "std_cf": float(v.std()),
                              "min_cf": float(v.min())} if len(v) else
                             {"n": 0, "mean_cf": 0.0, "std_cf": 0.0, "min_cf": 0.0})
    return out


def run_multigen(rng=None, seed=None, cfg=DEFAULT, return_state=False):
    if rng is None:
        rng = np.random.default_rng(seed)
    N = cfg.N
    base = dict(BASE_PARAMS, max_membranes=cfg.max_membranes, thresh_2=cfg.thresh_2, thresh_3=cfg.thresh_3)
    calibrator = CalibratorAgent(base)
    params = dict(base)

    rhos, levels, types, gens_born = [], [], [], []
    for _ in range(cfg.n_initial):
        phase = 0.10 * np.arange(N) + 0.22 * rng.normal(size=N)
        rhos.append(pure_rho(np.exp(1j * phase)))
        levels.append(int(rng.integers(2, 5)))
        types.append("founder")
        gens_born.append(0)
    rhos = np.stack(rhos)
    protection = [True] * len(rhos)

    births_2 = births_3 = 0
    mirror_gain = []
    history = []
    max_pop = len(rhos)
    post_mirror_strength = lambda: params["post_birth_concentrate"] if cfg.mirror else 0.0

    for gen in range(cfg.n_generations):
        if len(rhos) >= params["max_membranes"]:
            break
        rhos = evolve(rhos, levels, cfg)
        if cfg.mirror:
            rhos = crystallize(rhos, cfg.eureka_str)

        n = len(rhos)
        F = fidelity_matrix(rhos)
        cap = params["max_membranes"]
        kids, kid_levels, kid_types = [], [], []
        for i, j in combinations(range(n), 2):
            if n + len(kids) >= cap:
                break
            if F[i, j] >= params["thresh_2"] and rng.random() < params["fecund_p2"]:
                m = params["mix_2"]
                kids.append((1 - m) * rhos[i] + m * rhos[j])
                kid_levels.append(max(levels[i], levels[j]) + 1)
                kid_types.append("binary")
                births_2 += 1
        if n >= 3 and n + len(kids) < cap:
            # Vectorized pre-filter: only triples passing thresh_3 can draw from
            # the RNG or give birth, so iterating just those (in the same
            # lexicographic order) is equivalent to the full combinations loop.
            T = _triples(n)
            r3 = (F[T[:, 0], T[:, 1]] * F[T[:, 1], T[:, 2]] * F[T[:, 2], T[:, 0]]) ** (1.0 / 3.0)
            for i, j, k in T[r3 >= params["thresh_3"]].tolist():
                if n + len(kids) >= cap:
                    break
                if rng.random() < params["fecund_p3"]:
                    m = params["mix_3"]
                    kids.append((1 - m) * rhos[i] + (m / 2) * rhos[j] + (m / 2) * rhos[k])
                    kid_levels.append(max(levels[i], levels[j], levels[k]) + 1)
                    kid_types.append("ternary")
                    births_3 += 1
        if kids:
            raw = np.stack(kids)
            out = crystallize(raw, post_mirror_strength())
            mirror_gain.extend((purity(out) - purity(raw)).tolist())
            rhos = np.concatenate([rhos, out])
            levels += kid_levels
            types += kid_types
            gens_born += [gen + 1] * len(kids)
            protection += [True] * len(kids)
        max_pop = max(max_pop, len(rhos))
        rep = _observe(rhos, types, protection)
        rep["generation_index"] = gen
        params = calibrator.act(rep)
        history.append({"generation": gen, "n": rep["n"], "mean_cf": rep["mean_cf"],
                        "min_cf": rep["min_cf"], "births": len(kids),
                        "mirror": params["post_birth_concentrate"]})

    fin = _observe(rhos, types, protection)
    ev_min = float(np.linalg.eigvalsh(herm(rhos)).min())
    res = {
        "final_n": fin["n"], "births_2": births_2, "births_3": births_3,
        "mean_cf": fin["mean_cf"], "std_cf": fin["std_cf"], "min_cf": fin["min_cf"],
        "prot": fin["protection_fraction"], "gens": len(history),
        "bin_mean": fin["by_type"]["binary"]["mean_cf"], "bin_std": fin["by_type"]["binary"]["std_cf"],
        "ter_mean": fin["by_type"]["ternary"]["mean_cf"], "ter_std": fin["by_type"]["ternary"]["std_cf"],
        "found_mean": fin["by_type"]["founder"]["mean_cf"],
        "mean_purity": fin["mean_purity"], "min_purity": fin["min_purity"],
        "mean_entropy": float(entropy_normalized(rhos).mean()),
        "mirror_gain": float(np.mean(mirror_gain)) if mirror_gain else 0.0,
        "max_trace_err": float(np.abs(np.trace(rhos, axis1=-2, axis2=-1) - 1).max()),
        "max_herm_err": float(np.abs(rhos - np.conj(np.swapaxes(rhos, -1, -2))).max()),
        "min_eig": ev_min,
        "max_pop": max_pop,
        "history": history,
    }
    if return_state:
        res["rhos"] = rhos
    return res


def monte_carlo(n_samples=DEFAULT.n_samples, seed=DEFAULT.seed, cfg=DEFAULT, verbose=True):
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 68)
    log("PHMT-4 v2  |  density matrices + Lindblad dephasing + mirror")
    log("=" * 68)
    log(f"Samples {n_samples} | gens {cfg.n_generations} | start {cfg.n_initial} | "
        f"cap {cfg.max_membranes} | dephasing {cfg.dephasing:g} | seed {seed} | mirror {'on' if cfg.mirror else 'OFF (ablation)'}")
    rows = []
    for i in range(n_samples):
        r = run_multigen(rng, cfg=cfg)
        rows.append({k: v for k, v in r.items() if k != "history"})
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{n_samples}")
    df = pd.DataFrame(rows)
    log("\nRESULTS (SYNTHETIC)")
    for label, col, fmt in [
        ("Mean final n", "final_n", ".2f"), ("Mean binary births", "births_2", ".2f"),
        ("Mean ternary births", "births_3", ".2f"), ("Mean final Cf", "mean_cf", ".3f"),
        ("Mean std Cf", "std_cf", ".3f"), ("Mean min Cf", "min_cf", ".3f"),
        ("Mean founder Cf", "found_mean", ".3f"), ("Mean binary-offspring Cf", "bin_mean", ".3f"),
        ("Mean ternary-offspring Cf", "ter_mean", ".3f"), ("Mean purity", "mean_purity", ".4f"),
        ("Mean min purity", "min_purity", ".4f"), ("Mean entropy S/lnN", "mean_entropy", ".4f"),
        ("Mean mirror purity gain", "mirror_gain", ".2e"),
        ("Max |tr rho - 1|", "max_trace_err", ".1e"), ("Max Hermiticity err", "max_herm_err", ".1e"),
        ("Min eigenvalue", "min_eig", ".1e"), ("Mean generations", "gens", ".1f"),
    ]:
        agg = df[col].max() if col.startswith("max_") else (df[col].min() if col == "min_eig" else df[col].mean())
        log(f"{label:<29}: {agg:{fmt}}")
    log(f"{'Protection held':<29}: {100 * df.prot.mean():.1f}%")
    log("Scope: heuristic classical simulation. Not a consciousness theory.")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="PHMT-4 Monte Carlo v2 (density-matrix heuristic simulation)")
    ap.add_argument("--samples", type=int, default=DEFAULT.n_samples)
    ap.add_argument("--seed", type=int, default=DEFAULT.seed)
    ap.add_argument("--no-mirror", action="store_true", help="ablation: disable every crystallization")
    ap.add_argument("--dephasing", type=float, default=DEFAULT.dephasing, help="dephasing rate gamma (default 0.05)")
    ap.add_argument("--cap", type=int, default=DEFAULT.max_membranes, help="population cap (default 48)")
    ap.add_argument("--thresh2", type=float, default=DEFAULT.thresh_2, help="binary resonance threshold (default 0.70)")
    ap.add_argument("--thresh3", type=float, default=DEFAULT.thresh_3, help="ternary resonance threshold (default 0.76)")
    ap.add_argument("--csv", help="write per-sample rows to this CSV file")
    ap.add_argument("--json", help="write a summary (means, runtime, config) to this JSON file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(DEFAULT, mirror=not args.no_mirror, dephasing=args.dephasing, max_membranes=args.cap,
                  thresh_2=args.thresh2, thresh_3=args.thresh3)
    t0 = time.perf_counter()
    df = monte_carlo(args.samples, seed=args.seed, cfg=cfg, verbose=not args.quiet)
    runtime = time.perf_counter() - t0
    if not args.quiet:
        print(f"Runtime: {runtime:.2f} s (SYNTHETIC run, this machine)")
    if args.csv:
        df.to_csv(args.csv, index=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"version": "v2", "evidence_tag": "SYNTHETIC", "samples": args.samples,
                       "seed": args.seed, "runtime_s": round(runtime, 3), "config": asdict(cfg),
                       "means": {k: float(v) for k, v in df.mean(numeric_only=True).items()}},
                      f, indent=2)
    return df


if __name__ == "__main__":
    main()
