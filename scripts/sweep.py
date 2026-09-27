"""Parameter sweep for PHMT-4 v2: dephasing rate x population cap x mirror on/off.

Every number written is SYNTHETIC (a real run of this code). Each setting runs
``--samples`` independent runs from one Generator seeded with ``--seed`` (the
same seed for every setting). Settings run in parallel worker processes, each
pinned to one BLAS thread, so the per-setting runtime is single-threaded wall
time and not directly comparable with the multi-threaded runtimes elsewhere.

Run: python scripts/sweep.py                       # default grid, 40 samples, seed 42
     python scripts/sweep.py --dephasing 0.05 1 --caps 48 --samples 5 --out /tmp/s.csv
     python scripts/sweep.py --thresh2 0.80 0.85 0.88 0.90 --dephasing 0.05 0.5 2.0 --caps 96 \
         --out results/sweep_thresholds_seed42.csv   # follow-up: threshold lever
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phmt4.v2 as P  # noqa: E402


def run_setting(args):
    dephasing, cap, mirror, samples, seed, t2 = args
    t3 = round(min(0.92, t2 + 0.06), 4)
    cfg = replace(P.DEFAULT, dephasing=dephasing, max_membranes=cap, mirror=mirror, thresh_2=t2, thresh_3=t3)
    t0 = time.perf_counter()
    df = P.monte_carlo(samples, seed=seed, cfg=cfg, verbose=False)
    runtime = time.perf_counter() - t0
    return {
        "dephasing": dephasing, "cap": cap, "thresh_2": t2, "thresh_3": t3, "mirror": "on" if mirror else "off",
        "samples": samples, "seed": seed,
        "gens_mean": df.gens.mean(),
        "frac_all_gens": (df.gens == cfg.n_generations).mean(),
        "frac_cap_hit": (df.final_n >= cap).mean(),
        "final_n_mean": df.final_n.mean(),
        "births_2_mean": df.births_2.mean(), "births_3_mean": df.births_3.mean(),
        "mean_cf": df.mean_cf.mean(), "min_cf": df.min_cf.mean(),
        "mean_purity": df.mean_purity.mean(), "min_purity": df.min_purity.mean(),
        "mean_entropy": df.mean_entropy.mean(),
        "mirror_gain": df.mirror_gain.mean(),
        "max_trace_err": df.max_trace_err.max(), "min_eig": df.min_eig.min(),
        "runtime_s": runtime,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dephasing", type=float, nargs="+", default=[0.05, 0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--caps", type=int, nargs="+", default=[48, 96, 192])
    ap.add_argument("--thresh2", type=float, nargs="+", default=[0.70],
                    help="binary thresholds to sweep; thresh_3 = min(0.92, thresh_2 + 0.06) (0.70 -> 0.76 = default)")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "sweep_seed42.csv"))
    a = ap.parse_args(argv)
    jobs = [(d, c, m, a.samples, a.seed, t) for t in a.thresh2 for d in a.dephasing for c in a.caps for m in (True, False)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_setting, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["dephasing", "cap", "thresh_2", "mirror", "gens_mean", "frac_all_gens", "final_n_mean", "births_2_mean",
                  "births_3_mean", "mean_cf", "min_cf", "mean_purity", "mirror_gain", "runtime_s"]].round(4))
    print(f"wrote {a.out}  ({len(df)} settings, total wall {time.perf_counter() - t0:.1f} s, SYNTHETIC)")


if __name__ == "__main__":
    main()
