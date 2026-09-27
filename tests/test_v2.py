from dataclasses import replace

import numpy as np
import pytest

import phmt4.v2 as P

N = P.DEFAULT.N


def rand_rho(rng, rank=4):
    X = rng.normal(size=(N, rank)) + 1j * rng.normal(size=(N, rank))
    R = X @ X.conj().T
    return R / np.trace(R).real


def rand_pure(rng):
    return P.pure_rho(rng.normal(size=N) + 1j * rng.normal(size=N))


def assert_density(rho, tol=1e-10):
    rho = np.asarray(rho)
    assert np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0, atol=tol)
    assert np.allclose(rho, np.conj(np.swapaxes(rho, -1, -2)), atol=tol)
    assert np.linalg.eigvalsh(P.herm(rho)).min() > -tol


def test_pure_rho_is_density():
    assert_density(rand_pure(np.random.default_rng(0)))


def test_coherent_fraction_bounds_and_extremes():
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert 0.0 <= P.coherent_fraction(rand_rho(rng)) <= 1.0
    assert P.coherent_fraction(P.pure_rho(np.exp(1j * rng.uniform(0, 6.3, N)))) == pytest.approx(1.0)
    assert P.coherent_fraction(np.eye(N) / N) == pytest.approx(0.0)


def test_resonance_is_fidelity_symmetric_bounded_and_matches_pure_overlap():
    rng = np.random.default_rng(2)
    for _ in range(10):
        a, b = rand_rho(rng), rand_rho(rng)
        f = P.resonance(a, b)
        assert 0.0 <= f <= 1.0
        assert f == pytest.approx(P.resonance(b, a), abs=1e-8)
    a = rng.normal(size=N) + 1j * rng.normal(size=N)
    b = rng.normal(size=N) + 1j * rng.normal(size=N)
    ov = abs(np.vdot(a / np.linalg.norm(a), b / np.linalg.norm(b))) ** 2
    assert P.resonance(P.pure_rho(a), P.pure_rho(b)) == pytest.approx(ov, abs=1e-6)
    r = rand_rho(rng)
    assert P.resonance(r, r) == pytest.approx(1.0, abs=1e-8)


def test_crystallize_valid_identity_on_pure_and_sharpens_mixed():
    rng = np.random.default_rng(3)
    pure = rand_pure(rng)
    assert np.allclose(P.crystallize(pure, 0.58), pure, atol=1e-10)
    mixed = rand_rho(rng)
    out = P.crystallize(mixed, 0.58)
    assert_density(out)
    assert P.purity(out) > P.purity(mixed) + 1e-3


def test_evolve_preserves_trace_hermiticity_positivity():
    rng = np.random.default_rng(4)
    rhos = np.stack([rand_rho(rng) for _ in range(5)] + [rand_pure(rng)])
    out = P.evolve(rhos, [2, 3, 4, 5, 6, 3])
    assert_density(out)
    # dephasing without the mirror must not increase purity
    cfg = replace(P.DEFAULT, mirror=False)
    out2 = P.evolve(rhos, [2, 3, 4, 5, 6, 3], cfg)
    assert_density(out2)
    assert (P.purity(out2) <= P.purity(rhos) + 1e-10).all()


def test_unitary_step_is_unitary():
    U = P.unitary(3, P.DEFAULT.dt, N)
    assert np.allclose(U @ U.conj().T, np.eye(N), atol=1e-12)


def test_strang_step_is_second_order():
    rng = np.random.default_rng(5)
    rho0 = rand_rho(rng)
    T = 0.4

    def integrate(dt):
        cfg = replace(P.DEFAULT, dt=dt, steps_per_gen=int(round(T / dt)), mirror=False)
        return P.evolve(rho0[None], [3], cfg)[0]

    ref = integrate(T / 3200)
    e1 = np.abs(integrate(T / 20) - ref).max()
    e2 = np.abs(integrate(T / 40) - ref).max()
    assert e1 / e2 == pytest.approx(4.0, rel=0.25)


def test_calibrator_clipping_bounds():
    cal = P.CalibratorAgent(P.BASE_PARAMS)
    bad = {"n": 48, "min_cf": 0.0, "std_cf": 0.5, "mean_cf": 0.1, "generation_index": 0,
           "by_type": {"ternary": {"n": 3, "min_cf": 0.0}}}
    good = {"n": 2, "min_cf": 0.99, "std_cf": 0.0, "mean_cf": 0.99, "generation_index": 0, "by_type": {}}
    for rep in [bad] * 30 + [good] * 60:
        p = cal.act(rep)
        for k, (lo, hi) in P.CalibratorAgent.CLIP.items():
            assert lo <= p[k] <= hi, k


def test_run_deterministic_valid_protected_and_capped():
    a = P.run_multigen(seed=7, return_state=True)
    b = P.run_multigen(seed=7, return_state=True)
    assert np.array_equal(a["rhos"], b["rhos"])
    for k in a:
        if k not in ("rhos", "history"):
            assert a[k] == b[k], k
    assert_density(a["rhos"])
    assert a["prot"] == 1.0
    assert a["max_pop"] <= P.DEFAULT.max_membranes and a["final_n"] <= P.DEFAULT.max_membranes


@pytest.mark.parametrize("cap", [7, 13])
def test_cap_respected_with_small_cap(cap):
    cfg = replace(P.DEFAULT, max_membranes=cap)
    for s in range(3):
        r = P.run_multigen(seed=s, cfg=cfg)
        assert r["max_pop"] <= cap and r["final_n"] <= cap


def test_mirror_has_measurable_effect():
    on = P.run_multigen(seed=3)
    off = P.run_multigen(seed=3, cfg=replace(P.DEFAULT, mirror=False))
    assert on["mirror_gain"] > 0 and off["mirror_gain"] == 0
    assert on["mean_purity"] > off["mean_purity"]


def test_small_monte_carlo_reproducible(tmp_path):
    d1 = P.monte_carlo(3, seed=42, verbose=False)
    d2 = P.monte_carlo(3, seed=42, verbose=False)
    assert len(d1) == 3 and d1.equals(d2)
    assert (d1.prot == 1.0).all()


def test_cli_writes_csv_and_json(tmp_path):
    import json
    csv, js = tmp_path / "r.csv", tmp_path / "r.json"
    P.main(["--samples", "2", "--seed", "1", "--quiet", "--csv", str(csv), "--json", str(js)])
    assert csv.read_text().count("\n") == 3
    d = json.loads(js.read_text())
    assert d["evidence_tag"] == "SYNTHETIC" and d["samples"] == 2
