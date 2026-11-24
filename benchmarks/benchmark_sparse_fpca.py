"""Benchmark of the sparse FPCA pipeline (estimation, scoring, regression).

Runs the sparse covariance estimation, the conditional FPCA scores and
the FPCA regression on simulated sparse datasets of 300 and 3000 curves
with at most 8 observations per curve and an 81-point estimation grid,
and reports wall times and peak memory of each stage.

Each stage is executed in a subprocess (``--stage``/``--n-curves``
options) so that the peak resident memory (``ru_maxrss``) is attributed
to that stage only; without arguments, all combinations are run and
collected into a table.

The pipeline never builds dense matrices over all pairs of observations:
the largest intermediate objects are the per-curve observation
covariances (at most 8x8), the fixed-size grid operators (81x81) and
spline Gram matrices (bounded by the basis size). The state that grows
with the number of curves N is the input data O(N), the within-curve
cross-product arrays O(N) (three one-dimensional arrays; their large
design matrices are accumulated in fixed-size chunks of
O(chunk_size * cov_n_basis^2)) and the score output
O(N * n_components).

Usage:
    python benchmarks/benchmark_sparse_fpca.py
    python benchmarks/benchmark_sparse_fpca.py --n-curves 300 --stage estimate
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np

GRID = np.linspace(0.0, 1.0, 81)
DOMAIN = [(0.0, 1.0)]
LAM = np.array([1.0, 0.5, 0.25])
SIGMA2 = 0.04
N_COMPONENTS = 3
MAX_POINTS = 8


def _mu(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * t


def _phis(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.stack(
        [
            np.sqrt(2.0) * np.sin(2 * np.pi * t),
            np.sqrt(2.0) * np.cos(2 * np.pi * t),
            np.sqrt(2.0) * np.sin(4 * np.pi * t),
        ],
        axis=-1,
    )


def _make_curves(n_curves: int, seed: int):
    from skfda.representation import FDataIrregular

    rng = np.random.default_rng(seed)
    points = []
    values = []
    start_indices = [0]
    scores_all = []
    for _ in range(n_curves):
        n_points = rng.integers(2, MAX_POINTS + 1)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        scores = rng.normal(0.0, np.sqrt(LAM))
        latent = _mu(curve_points) + _phis(curve_points) @ scores
        observed = latent + rng.normal(0.0, np.sqrt(SIGMA2), n_points)
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)
        scores_all.append(scores)

    fd = FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=DOMAIN,
    )
    return fd, np.array(scores_all)


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_stage(stage: str, n_curves: int, seed: int) -> dict:
    """Run one benchmark stage and return timings and memory peaks."""
    from skfda.exploratory.stats.covariance import EmpiricalCovariance
    from skfda.ml.regression import FPCARegression
    from skfda.preprocessing.dim_reduction import FPCA

    train, scores_train = _make_curves(n_curves, seed)
    test, _ = _make_curves(n_curves, seed + 1)
    y_train = 3.0 + 2.0 * scores_train[:, 0] - scores_train[:, 1]

    n_obs = len(train.points)
    end_indices = np.concatenate((train.start_indices[1:], [n_obs]))
    n_pairs = int(
        np.sum(
            (end_indices - train.start_indices)
            * (end_indices - train.start_indices - 1),
        ),
    )
    shapes = {
        "pooled observations": f"({n_obs},)",
        "within-curve pairs (streamed)": f"({n_pairs},)",
        "pair design chunk": "(16384, cov_n_basis^2=100)",
        "spline Gram matrices": "(100, 100) fixed",
        "grid covariance operator": "(81, 81) fixed",
        "per-curve observation covariance": "(<=8, <=8)",
        "scores output": f"({n_curves}, {N_COMPONENTS})",
    }

    result = {"stage": stage, "n_curves": n_curves, "shapes": shapes}

    tracemalloc.start()
    if stage == "estimate":
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        start = time.perf_counter()
        estimator.fit(train)
        result["time"] = time.perf_counter() - start
        result["noise_variance"] = estimator.noise_variance_

    elif stage == "score":
        fpca = FPCA(
            n_components=N_COMPONENTS,
            covariance_estimator=EmpiricalCovariance(
                sparse=True,
                grid_points=GRID,
            ),
        )
        start = time.perf_counter()
        fpca.fit(train)
        result["time_fit"] = time.perf_counter() - start
        start = time.perf_counter()
        scores = fpca.transform(test)
        result["time"] = time.perf_counter() - start
        assert scores.shape == (n_curves, N_COMPONENTS)

    elif stage == "regress":
        regression = FPCARegression(
            n_components=N_COMPONENTS,
            covariance_estimator=EmpiricalCovariance(
                sparse=True,
                grid_points=GRID,
            ),
        )
        start = time.perf_counter()
        regression.fit(train, y_train)
        result["time_fit"] = time.perf_counter() - start
        start = time.perf_counter()
        predictions = regression.predict(test)
        result["time"] = time.perf_counter() - start
        assert predictions.shape == (n_curves,)

    else:
        raise ValueError(f"Unknown stage {stage!r}")

    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_tracemalloc_mb"] = peak_tracemalloc / 2**20
    result["peak_rss_mb"] = _peak_rss_mb()
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-curves", type=int, choices=(300, 3000))
    parser.add_argument(
        "--stage",
        choices=("estimate", "score", "regress"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.stage is not None and args.n_curves is not None:
        # Subprocess mode: run a single stage and print it as JSON.
        result = _run_stage(args.stage, args.n_curves, args.seed)
        print(json.dumps(result))
        return

    # Parent mode: run every combination in its own subprocess.
    rows = []
    for n_curves in (300, 3000):
        for stage in ("estimate", "score", "regress"):
            completed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--stage",
                    stage,
                    "--n-curves",
                    str(n_curves),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            rows.append(json.loads(completed.stdout))

    header = (
        f"{'curves':>6} {'stage':>8} | {'time (s)':>9} {'fit (s)':>8} | "
        f"{'peak RSS (MiB)':>14} {'peak pymalloc (MiB)':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        fit_time = f"{row['time_fit']:8.3f}" if "time_fit" in row else " " * 8
        print(
            f"{row['n_curves']:>6} {row['stage']:>8} | "
            f"{row['time']:>9.3f} {fit_time} | "
            f"{row['peak_rss_mb']:>14.1f} {row['peak_tracemalloc_mb']:>18.1f}",
        )

    print("\nMain intermediate shapes (3000 curves):")
    for name, shape in rows[-1]["shapes"].items():
        print(f"  {name}: {shape}")

    print(
        "\nNotes: 'time' is the stage proper (estimator fit, transform "
        "of a held-out batch of the same size, or prediction), 'fit' is "
        "the preceding model fit when the stage needs one. Peak RSS is "
        "the high-water mark of the subprocess. No dense matrix over all "
        "pairs of observations is ever built: the N-dependent state is "
        "the input data, the within-curve pair arrays and the "
        "score/prediction outputs (all linear in N); the large pair "
        "design matrices are accumulated in fixed-size chunks, and the "
        "grid operators and spline Gram matrices have fixed size.",
    )


if __name__ == "__main__":
    _main()
