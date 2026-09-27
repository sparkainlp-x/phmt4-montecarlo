import numpy as np
import pytest

import phmt4.v1 as P


def rand_psi(rng):
    return P.normalize(rng.normal(size=P.N) + 1j * rng.normal(size=P.N))


def test_normalize_unit_and_zero_fallback():
    rng = np.random.default_rng(1)
    assert np.isclose(np.linalg.norm(P.normalize(rng.normal(size=P.N) * 7)), 1.0)
    z = P.normalize(np.zeros(P.N, dtype=complex))
    assert np.isclose(np.linalg.norm(z), 1.0)


def test_resonance_symmetric_and_bounded():
    rng = np.random.default_rng(2)
    for _ in range(50):
        a, b = rand_psi(rng), rand_psi(rng)
        r = P.resonance(a, b)
        assert 0.0 <= r <= 1.0 + 1e-12
        assert r == pytest.approx(P.resonance(b, a), abs=1e-15)
    a = rand_psi(rng)
    assert P.resonance(a, a) == pytest.approx(1.0)


def test_coherent_fraction_bounds_and_extremes():
    rng = np.random.default_rng(3)
    for _ in range(20):
        cf = P.coherent_fraction(P.to_rho(rand_psi(rng)))
        assert 0.0 <= cf <= 1.0
    # uniform magnitude => Cf = 1, whatever the phases
    for phases in (np.zeros(P.N), rng.uniform(0, 2 * np.pi, P.N)):
        assert P.coherent_fraction(P.to_rho(np.exp(1j * phases))) == pytest.approx(1.0)
    e = np.zeros(P.N, dtype=complex)
    e[5] = 1
    assert P.coherent_fraction(P.to_rho(e)) == pytest.approx(0.0)


def test_crystallize_unit_vector_and_pure_state_limitation():
    rng = np.random.default_rng(4)
    for s in (0.0, 0.35, 0.58, 0.9):
        psi = rand_psi(rng)
        c = P.crystallize(psi, s)
        assert np.linalg.norm(c) == pytest.approx(1.0)
        # documented limitation: on pure states the mirror is a global phase
        assert P.resonance(psi, c) == pytest.approx(1.0, abs=1e-12)


def test_calibrator_clipping_bounds():
    cal = P.CalibratorAgent(P.BASE_PARAMS)
    bad = {"n": 48, "min_cf": 0.0, "std_cf": 0.5, "mean_cf": 0.1, "generation_index": 0,
           "by_type": {"ternary": {"n": 3, "min_cf": 0.0}}}
    good = {"n": 2, "min_cf": 0.99, "std_cf": 0.0, "mean_cf": 0.99, "generation_index": 0, "by_type": {}}
    for rep in [bad] * 30 + [good] * 60:
        p = cal.act(rep)
        for k, (lo, hi) in P.CalibratorAgent.CLIP.items():
            assert lo <= p[k] <= hi, k


def test_run_multigen_deterministic_protection_and_cap():
    a = P.run_multigen(seed=7)
    b = P.run_multigen(seed=7)
    for k in a:
        if k != "history":
            assert a[k] == b[k], k
    assert a["history"] == b["history"]
    assert a["prot"] == 1.0
    assert a["final_n"] <= P.C.max_membranes
    assert a["max_pop"] <= P.C.max_membranes


def test_cap_respected_with_small_cap(monkeypatch):
    monkeypatch.setattr(P.C, "max_membranes", 9)
    monkeypatch.setitem(P.BASE_PARAMS, "max_membranes", 9)
    for s in range(3):
        r = P.run_multigen(seed=s)
        assert r["max_pop"] <= 9 and r["final_n"] <= 9


def test_small_monte_carlo_is_reproducible():
    d1 = P.monte_carlo(3, seed=42, verbose=False)
    d2 = P.monte_carlo(3, seed=42, verbose=False)
    assert len(d1) == 3
    assert d1.equals(d2)
    assert (d1.prot == 1.0).all()


def test_v1_matches_original_bit_for_bit():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "original" / "phmt4_montecarlo_original.py"
    spec = importlib.util.spec_from_file_location("phmt4_original", path)
    O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(O)
    g1, g2 = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(3):
        a, b = O.run_multigen(g1), P.run_multigen(g2)
        for k in a:
            if k != "history":
                assert a[k] == b[k], k
