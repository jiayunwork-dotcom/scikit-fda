"""Tests for the sparse mode of EmpiricalCovariance."""
from __future__ import annotations

import pickle
import unittest

import numpy as np
import scipy.integrate
from sklearn.base import clone

from skfda.exploratory.stats.covariance import EmpiricalCovariance
from skfda.misc.covariances import EigenExpansion, EmpiricalGrid
from skfda.representation import FDataGrid, FDataIrregular

GRID = np.linspace(0.0, 1.0, 81)
DOMAIN = [(0.0, 1.0)]

# True model used in the simulation tests: a smooth low-rank process
# X(t) = mu(t) + sum_k xi_k phi_k(t), xi_k ~ N(0, LAM[k]), observed with
# independent noise of variance SIGMA2.
LAM = np.array([1.0, 0.5, 0.25])
SIGMA2 = 0.04


def _mu(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * t


def _phis(t: np.ndarray) -> np.ndarray:
    """Evaluate the three true eigenfunctions (columns) at t."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack(
        [
            np.sqrt(2.0) * np.sin(2 * np.pi * t),
            np.sqrt(2.0) * np.cos(2 * np.pi * t),
            np.sqrt(2.0) * np.sin(4 * np.pi * t),
        ],
        axis=-1,
    )


K_TRUE = sum(
    lam * np.outer(phi, phi)
    for lam, phi in zip(LAM, _phis(GRID).T)
)


def _make_sparse_sample(
    n_curves: int,
    seed: int,
    *,
    noise: float = SIGMA2,
) -> FDataIrregular:
    """Simulate sparse noisy observations of the true process."""
    rng = np.random.default_rng(seed)
    points = []
    values = []
    start_indices = [0]
    for _ in range(n_curves):
        n_points = rng.integers(2, 9)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        scores = rng.normal(0.0, np.sqrt(LAM))
        latent = _mu(curve_points) + _phis(curve_points) @ scores
        observed = latent + rng.normal(0.0, np.sqrt(noise), n_points)
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)

    return FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=DOMAIN,
    )


def _surface_l2_error(estimated: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error between two covariance surfaces on GRID."""
    numerator = scipy.integrate.simpson(
        scipy.integrate.simpson((estimated - true) ** 2, x=GRID, axis=1),
        x=GRID,
    )
    denominator = scipy.integrate.simpson(
        scipy.integrate.simpson(true ** 2, x=GRID, axis=1),
        x=GRID,
    )
    return float(np.sqrt(numerator / denominator))


def _subspace_error(
    estimated: np.ndarray,
    true: np.ndarray,
    n_components: int = 3,
) -> float:
    """Sine of the largest principal angle between principal subspaces.

    The subspaces are those spanned by the first eigenfunctions of the
    discretized covariance operators (with the integration weights of
    the grid).
    """
    weights = scipy.integrate.simpson(np.eye(len(GRID)), x=GRID)
    sqrt_weights = np.sqrt(weights)

    def top_eigenvectors(cov: np.ndarray) -> np.ndarray:
        operator = sqrt_weights[:, None] * cov * sqrt_weights[None, :]
        _, eigenvectors = np.linalg.eigh(operator)
        return eigenvectors[:, ::-1][:, :n_components]

    first = top_eigenvectors(estimated)
    second = top_eigenvectors(true)
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - singular_values[-1] ** 2)))


def _interpolated_covariance(fd: FDataIrregular) -> np.ndarray:
    """Baseline: per-curve linear interpolation + dense empirical cov."""
    data_matrix = np.empty((fd.n_samples, len(GRID)))
    end_indices = np.concatenate((fd.start_indices[1:], [len(fd.points)]))
    for curve, (start, end) in enumerate(zip(fd.start_indices, end_indices)):
        data_matrix[curve] = np.interp(
            GRID,
            fd.points[start:end, 0],
            fd.values[start:end, 0],
        )
    fd_grid = FDataGrid(
        data_matrix[:, :, np.newaxis],
        grid_points=GRID,
        domain_range=DOMAIN,
    )
    estimator = EmpiricalCovariance()
    estimator.fit(fd_grid)
    return estimator.covariance_(GRID, GRID)


class SparseEmpiricalCovarianceValidationTestCase(unittest.TestCase):
    """Validation of the inputs of the sparse estimator."""

    def test_default_mode_is_unchanged(self) -> None:
        """The default estimator keeps returning the sample covariance."""
        fd = FDataGrid(
            np.arange(12, dtype=float).reshape(3, 4, 1),
            grid_points=[0.0, 1.0, 2.0, 3.0],
        )
        estimator = EmpiricalCovariance()
        self.assertFalse(estimator.sparse)
        estimator.fit(fd)
        self.assertIsInstance(estimator.covariance_, EmpiricalGrid)
        np.testing.assert_allclose(
            estimator.location_.data_matrix,
            fd.mean().data_matrix,
        )
        self.assertFalse(hasattr(estimator, "noise_variance_"))

    def test_fit_requires_irregular_input(self) -> None:
        """sparse=True with dense input raises a ValueError."""
        fd = FDataGrid(
            np.arange(12, dtype=float).reshape(3, 4, 1),
            grid_points=[0.0, 1.0, 2.0, 3.0],
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_fit_rejects_multidimensional_codomain(self) -> None:
        """Vector-valued curves are rejected."""
        fd = FDataIrregular(
            start_indices=[0, 2, 4, 6],
            points=[[0.1], [0.2], [0.1], [0.2], [0.1], [0.2]],
            values=[
                [1.0, 0.0], [2.0, 1.0], [1.0, 0.0],
                [2.0, 1.0], [1.0, 0.0], [2.0, 1.0],
            ],
            domain_range=DOMAIN,
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(NotImplementedError):
            estimator.fit(fd)

    def test_fit_rejects_multidimensional_domain(self) -> None:
        """Curves with a bidimensional domain are rejected."""
        fd = FDataIrregular(
            start_indices=[0, 2, 4, 6],
            points=[
                [0.1, 0.0], [0.2, 0.0], [0.1, 0.0],
                [0.2, 0.0], [0.1, 0.0], [0.2, 0.0],
            ],
            values=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            domain_range=[(0.0, 1.0), (0.0, 1.0)],
        )
        estimator = EmpiricalCovariance(
            sparse=True,
            grid_points=[0.0, 0.5, 1.0],
        )
        with self.assertRaises(NotImplementedError):
            estimator.fit(fd)

    def test_fit_requires_three_curves(self) -> None:
        """Fewer than three curves are rejected."""
        fd = _make_sparse_sample(2, seed=0)
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_fit_requires_two_observations_per_curve(self) -> None:
        """A curve with a single observation is rejected."""
        fd = FDataIrregular(
            start_indices=[0, 2, 3, 5],
            points=[[0.1], [0.2], [0.3], [0.1], [0.2]],
            values=[1.0, 2.0, 3.0, 1.0, 2.0],
            domain_range=DOMAIN,
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_fit_rejects_non_finite_observations(self) -> None:
        """NaN and infinite values or points are rejected."""
        base = _make_sparse_sample(5, seed=0)
        for bad_value in (np.nan, np.inf):
            values = base.values.copy()
            values[3, 0] = bad_value
            fd = FDataIrregular(
                start_indices=base.start_indices,
                points=base.points,
                values=values,
                domain_range=DOMAIN,
            )
            estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
            with self.assertRaises(ValueError):
                estimator.fit(fd)

        points = base.points.copy()
        points[2, 0] = np.nan
        fd = FDataIrregular(
            start_indices=base.start_indices,
            points=points,
            values=base.values,
            domain_range=DOMAIN,
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_fit_rejects_duplicated_times(self) -> None:
        """Duplicated observation times within a curve are rejected."""
        fd = FDataIrregular(
            start_indices=[0, 2, 4, 6],
            points=[[0.1], [0.1], [0.1], [0.2], [0.1], [0.2]],
            values=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            domain_range=DOMAIN,
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_fit_rejects_points_outside_domain(self) -> None:
        """Points outside the declared domain range are rejected."""
        fd = FDataIrregular(
            start_indices=[0, 2, 4, 6],
            points=[[0.1], [1.5], [0.1], [0.2], [0.1], [0.2]],
            values=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            domain_range=DOMAIN,
        )
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        with self.assertRaises(ValueError):
            estimator.fit(fd)

    def test_grid_points_validation(self) -> None:
        """Invalid evaluation grids are rejected."""
        fd = _make_sparse_sample(5, seed=0)

        # grid_points is required in sparse mode.
        with self.assertRaises(ValueError):
            EmpiricalCovariance(sparse=True).fit(fd)

        # At least 3 points are required.
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True,
                grid_points=[0.0, 1.0],
            ).fit(fd)

        # The grid must be strictly increasing.
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True,
                grid_points=[0.0, 0.5, 0.5, 1.0],
            ).fit(fd)

        # The grid must be finite.
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True,
                grid_points=[0.0, np.inf, 1.0],
            ).fit(fd)

        # The extremes must coincide with the domain range.
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True,
                grid_points=[0.0, 0.5, 0.9],
            ).fit(fd)

    def test_noise_variance_validation(self) -> None:
        """Invalid noise_variance values are rejected."""
        fd = _make_sparse_sample(5, seed=0)
        for bad_value in ("unknown", -0.5, np.nan, np.inf, True, [0.1]):
            with self.assertRaises(ValueError, msg=f"noise={bad_value}"):
                EmpiricalCovariance(
                    sparse=True,
                    grid_points=GRID,
                    noise_variance=bad_value,
                ).fit(fd)

        # Valid values are accepted.
        for good_value in ("auto", 0.0, 0.1, np.float64(0.2)):
            EmpiricalCovariance(
                sparse=True,
                grid_points=GRID,
                noise_variance=good_value,
            ).fit(fd)

    def test_smoothing_parameters_validation(self) -> None:
        """Invalid smoothing parameters are rejected."""
        fd = _make_sparse_sample(5, seed=0)
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True, grid_points=GRID, mean_n_basis=3,
            ).fit(fd)
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True, grid_points=GRID, cov_n_basis=2.5,
            ).fit(fd)
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True, grid_points=GRID, smoothing_parameter=-1.0,
            ).fit(fd)

    def test_insufficient_support_fails(self) -> None:
        """Observations covering only a subdomain make the fit fail.

        With no penalty, the B-spline basis functions supported on the
        unobserved region have an empty design and the normal equations
        are singular, so the estimator refuses to return a degenerate
        solution.
        """
        rng = np.random.default_rng(0)
        points = []
        values = []
        start_indices = [0]
        for _ in range(30):
            curve_points = np.sort(rng.uniform(0.0, 0.3, 5))
            points.append(curve_points)
            values.append(np.sin(curve_points))
            start_indices.append(start_indices[-1] + 5)
        fd = FDataIrregular(
            start_indices=np.array(start_indices[:-1]),
            points=np.concatenate(points)[:, np.newaxis],
            values=np.concatenate(values)[:, np.newaxis],
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            EmpiricalCovariance(
                sparse=True,
                grid_points=GRID,
                smoothing_parameter=0.0,
            ).fit(fd)


class SparseEmpiricalCovarianceEstimationTestCase(unittest.TestCase):
    """Estimation accuracy on simulated data with a known model.

    The true process is smooth and low-rank (three sinusoidal
    eigenfunctions) with independent noise of variance 0.04. Each curve
    is observed at 2 to 8 random times. The tolerances below were chosen
    from the empirical distribution of the errors over many seeds, with
    margin: they are loose enough to absorb the (large) variability of
    covariance and noise estimation with a few thousand sparse curves,
    and tight enough to fail if the estimator did not actually learn the
    structure of the process.
    """

    n_curves = 2000
    seeds = (0, 1, 2, 3, 4)

    def _fit(self, seed: int) -> EmpiricalCovariance:
        fd = _make_sparse_sample(self.n_curves, seed=seed)
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        estimator.fit(fd)
        return estimator

    def test_estimation_accuracy(self) -> None:
        """Mean, covariance, subspace and noise errors are bounded."""
        noise_estimates = []
        report = (
            f"{'seed':>4} {'noise':>8} {'cov_err':>8} "
            f"{'subsp_err':>9} {'mean_err':>8}"
        )
        for seed in self.seeds:
            estimator = self._fit(seed)

            mean_error = np.sqrt(scipy.integrate.simpson(
                (
                    estimator.location_.data_matrix[0, :, 0]
                    - _mu(GRID)
                ) ** 2,
                x=GRID,
            ))
            self.assertLess(mean_error, 0.15)

            cov_error = _surface_l2_error(
                estimator.covariance_(GRID, GRID),
                K_TRUE,
            )
            self.assertLess(cov_error, 0.2)

            subspace_error = _subspace_error(
                estimator.covariance_(GRID, GRID),
                K_TRUE,
            )
            self.assertLess(subspace_error, 0.25)

            noise_estimates.append(estimator.noise_variance_)
            self.assertGreaterEqual(estimator.noise_variance_, 0.0)
            self.assertLess(estimator.noise_variance_, 0.15)

            report += (
                f"\n{seed:>4} {estimator.noise_variance_:>8.4f} "
                f"{cov_error:>8.4f} {subspace_error:>9.4f} "
                f"{mean_error:>8.4f}"
            )

        # Reported metrics of the sparse estimation (visible with
        # pytest -s): noise estimate, covariance integrated error and
        # principal subspace error per seed.
        print(f"\nSparse estimation metrics (true noise {SIGMA2}):\n{report}")

        # The noise estimates fluctuate around the true value (their
        # variance is large with sparse curves), so the median over
        # seeds is required to be reasonably close to the true value.
        median_noise = np.median(noise_estimates)
        self.assertGreater(median_noise, 0.5 * SIGMA2)
        self.assertLess(median_noise, 2.5 * SIGMA2)

    def test_estimation_beats_interpolation_baseline(self) -> None:
        """The sparse estimator beats interpolation + dense covariance.

        The baseline interpolates each curve linearly onto the grid and
        computes the classical (dense) empirical covariance, which
        cannot separate the measurement noise and treats unobserved
        regions as observed linear pieces. The comparison is done on
        the median over the seeds, as a single seed can be unfavorable
        to either method.
        """
        sparse_errors = []
        baseline_errors = []
        for seed in self.seeds:
            fd = _make_sparse_sample(self.n_curves, seed=seed)
            estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
            estimator.fit(fd)
            sparse_errors.append(
                _surface_l2_error(estimator.covariance_(GRID, GRID), K_TRUE),
            )
            baseline_errors.append(
                _surface_l2_error(_interpolated_covariance(fd), K_TRUE),
            )

        self.assertLess(
            float(np.median(sparse_errors)),
            float(np.median(baseline_errors)),
        )

    def test_kernel_is_psd_off_grid(self) -> None:
        """The kernel is PSD on arbitrary in-domain point sets."""
        estimator = self._fit(seed=0)
        rng = np.random.default_rng(123)
        for _ in range(5):
            points = np.sort(rng.uniform(0.0, 1.0, 37))
            cov_matrix = estimator.covariance_(points, points)
            np.testing.assert_allclose(
                cov_matrix,
                cov_matrix.T,
                rtol=1e-10,
                atol=1e-12,
            )
            min_eigenvalue = np.linalg.eigvalsh(cov_matrix)[0]
            scale = np.abs(np.diag(cov_matrix)).max()
            self.assertGreaterEqual(min_eigenvalue, -1e-8 * scale)

    def test_kernel_evaluates_between_grid_points(self) -> None:
        """The kernel can be queried at points outside the grid."""
        estimator = self._fit(seed=0)
        points = GRID[:-1] + np.diff(GRID) / 2
        cov_matrix = estimator.covariance_(points, points)
        self.assertEqual(cov_matrix.shape, (len(points), len(points)))
        self.assertTrue(np.all(np.isfinite(cov_matrix)))
        # The true kernel is smooth, so the estimate at interior
        # midpoints is close to the true kernel there. The boundary
        # region is excluded, as the surface estimate is known to be
        # less reliable near the extremes of the domain.
        interior = (points >= 0.1) & (points <= 0.9)
        interior_points = points[interior]
        cov_interior = estimator.covariance_(interior_points, interior_points)
        k_true_mid = sum(
            lam * np.outer(phi, phi)
            for lam, phi in zip(LAM, _phis(interior_points).T)
        )
        np.testing.assert_allclose(
            cov_interior,
            k_true_mid,
            rtol=0,
            atol=0.35,
        )

    def test_kernel_query_errors(self) -> None:
        """Out-of-domain, non-finite and multidimensional queries fail."""
        estimator = self._fit(seed=0)
        kernel = estimator.covariance_

        with self.assertRaises(ValueError):
            kernel([-0.1, 0.5], [0.5])
        with self.assertRaises(ValueError):
            kernel([0.5], [1.2, 0.5])
        with self.assertRaises(ValueError):
            kernel([0.5, np.nan], [0.5])
        with self.assertRaises(NotImplementedError):
            kernel([[0.5, 0.6]], [[0.5, 0.6]])

    def test_fixed_noise_variance(self) -> None:
        """A provided noise variance is used instead of estimated."""
        fd = _make_sparse_sample(50, seed=0)
        estimator = EmpiricalCovariance(
            sparse=True,
            grid_points=GRID,
            noise_variance=0.07,
        )
        estimator.fit(fd)
        self.assertEqual(estimator.noise_variance_, 0.07)

        estimator_zero = EmpiricalCovariance(
            sparse=True,
            grid_points=GRID,
            noise_variance=0.0,
        )
        estimator_zero.fit(fd)
        self.assertEqual(estimator_zero.noise_variance_, 0.0)

    def test_assume_centered(self) -> None:
        """With assume_centered=True the location is zero."""
        fd = _make_sparse_sample(50, seed=0)
        estimator = EmpiricalCovariance(
            assume_centered=True,
            sparse=True,
            grid_points=GRID,
        )
        estimator.fit(fd)
        np.testing.assert_array_equal(
            estimator.location_.data_matrix,
            np.zeros((1, len(GRID), 1)),
        )

    def test_fit_does_not_modify_input(self) -> None:
        """The data arrays are not mutated by the fit."""
        fd = _make_sparse_sample(50, seed=0)
        points_before = fd.points.copy()
        values_before = fd.values.copy()
        start_before = fd.start_indices.copy()
        estimator = EmpiricalCovariance(sparse=True, grid_points=GRID)
        estimator.fit(fd)
        np.testing.assert_array_equal(fd.points, points_before)
        np.testing.assert_array_equal(fd.values, values_before)
        np.testing.assert_array_equal(fd.start_indices, start_before)

    def test_clone_and_params(self) -> None:
        """The estimator parameters are managed the sklearn way."""
        estimator = EmpiricalCovariance(
            sparse=True,
            grid_points=GRID,
            noise_variance=0.02,
            cov_n_basis=8,
        )
        cloned = clone(estimator)
        self.assertEqual(
            sorted(cloned.get_params()),
            sorted(estimator.get_params()),
        )
        np.testing.assert_array_equal(cloned.grid_points, GRID)
        self.assertEqual(cloned.noise_variance, 0.02)
        self.assertEqual(cloned.cov_n_basis, 8)

        cloned.set_params(noise_variance=0.03, mean_n_basis=6)
        self.assertEqual(cloned.noise_variance, 0.03)
        self.assertEqual(cloned.mean_n_basis, 6)
        # The original estimator is unchanged.
        self.assertEqual(estimator.noise_variance, 0.02)
        self.assertEqual(estimator.mean_n_basis, 12)

    def test_pickle_roundtrip(self) -> None:
        """A fitted estimator keeps its predictions after pickling."""
        estimator = self._fit(seed=0)
        points = np.linspace(0.0, 1.0, 23)
        restored = pickle.loads(pickle.dumps(estimator))
        np.testing.assert_allclose(
            restored.covariance_(points, points),
            estimator.covariance_(points, points),
        )
        np.testing.assert_allclose(
            restored.location_.data_matrix,
            estimator.location_.data_matrix,
        )
        self.assertEqual(
            restored.noise_variance_,
            estimator.noise_variance_,
        )

    def test_covariance_is_eigen_expansion(self) -> None:
        """The sparse covariance is delivered as an EigenExpansion."""
        estimator = self._fit(seed=0)
        self.assertIsInstance(estimator.covariance_, EigenExpansion)
        self.assertTrue(np.all(estimator.covariance_.eigenvalues >= 0))


if __name__ == "__main__":
    unittest.main()
