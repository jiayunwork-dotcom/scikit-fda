"""Tests for FPCA and FPCARegression with a covariance estimator."""
from __future__ import annotations

import pickle
import unittest

import numpy as np
import scipy.integrate
from sklearn.base import clone

from skfda.exploratory.stats.covariance import (
    CovarianceEstimator,
    EmpiricalCovariance,
)
from skfda.misc.covariances import EigenExpansion
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import FPCARegression
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation import FDataGrid, FDataIrregular
from skfda.representation.basis import FourierBasis

GRID = np.linspace(0.0, 1.0, 81)
DOMAIN = [(0.0, 1.0)]
TRUE_EIGENVALUES = (1.0, 0.5)


def _mu(t: np.ndarray) -> np.ndarray:
    """True mean function."""
    return 1.0 + 0.5 * t


def _true_eigenfunctions(t: np.ndarray) -> np.ndarray:
    """Evaluate the true eigenfunctions (columns) at t."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack(
        [
            np.sqrt(2.0) * np.sin(2 * np.pi * t),
            np.sqrt(2.0) * np.cos(2 * np.pi * t),
        ],
        axis=-1,
    )


def _true_kernel(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """True covariance kernel of the latent process."""
    s = np.asarray(s, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    phi_s = _true_eigenfunctions(s)
    phi_t = _true_eigenfunctions(t)
    return phi_s @ (np.asarray(TRUE_EIGENVALUES)[:, None] * phi_t.T)


class _KnownEstimator(CovarianceEstimator):
    """Covariance estimator with a known mean, kernel and noise.

    It ignores the data and always returns the same known model, so
    that the conditional computations of FPCA can be checked against
    independently computed Gaussian conditioning formulas.
    """

    def __init__(
        self,
        *,
        assume_centered: bool = False,
        noise_variance: float = 0.01,
        eigenvalues: tuple[float, ...] = TRUE_EIGENVALUES,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.noise_variance = noise_variance
        self.eigenvalues = eigenvalues

    def fit(  # noqa: D102
        self,
        X: FDataIrregular,
        y: object = None,
    ) -> _KnownEstimator:
        eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        eigenvectors = _true_eigenfunctions(GRID)[:, : len(eigenvalues)]
        self.location_ = FDataGrid(
            _mu(GRID)[np.newaxis, :, np.newaxis],
            grid_points=GRID,
            domain_range=DOMAIN,
        )
        self.covariance_ = EigenExpansion(
            grid_points=GRID,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        self.noise_variance_ = float(self.noise_variance)
        return self


class _EmptyEstimator(CovarianceEstimator):
    """Estimator that does not follow the required protocol."""

    def fit(self, X: FDataIrregular, y: object = None) -> _EmptyEstimator:  # noqa: D102
        return self


def _make_curves(
    n_curves: int,
    seed: int,
    *,
    noise: float = 0.01,
    min_points: int = 2,
    max_points: int = 8,
) -> tuple[FDataIrregular, np.ndarray]:
    """Simulate sparse noisy curves from the known process.

    Returns:
        The observed curves and the true latent scores of each curve.

    """
    rng = np.random.default_rng(seed)
    points = []
    values = []
    start_indices = [0]
    true_scores = []
    for _ in range(n_curves):
        n_points = rng.integers(min_points, max_points + 1)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        scores = rng.normal(0.0, np.sqrt(TRUE_EIGENVALUES))
        latent = (
            _mu(curve_points)
            + _true_eigenfunctions(curve_points) @ scores
        )
        observed = latent + rng.normal(0.0, np.sqrt(noise), n_points)
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)
        true_scores.append(scores)

    fd = FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=DOMAIN,
    )
    return fd, np.array(true_scores)


def _quadrature_weights() -> np.ndarray:
    """Integration weights of the grid, computed independently."""
    return scipy.integrate.simpson(np.eye(len(GRID)), x=GRID)


def _reference_scores(fpca: FPCA, fd: FDataIrregular) -> np.ndarray:
    """Independent Gaussian conditioning for the principal scores.

    This is a small, self-contained implementation of the conditional
    mean of the principal component scores given the noisy
    observations, E[xi_k | y] = lambda_k phi_k(t)' (K(t, t) +
    sigma^2 I)^{-1} (y - mu(t)), built only from public fitted
    attributes of the FPCA object and of its covariance estimator.
    """
    estimator = fpca.covariance_estimator_
    noise = estimator.noise_variance_
    eigenvalues = fpca.explained_variance_
    components = fpca.components_.data_matrix[..., 0]
    mean = fpca.mean_.data_matrix[0, :, 0]

    scores = np.empty((fd.n_samples, fpca.n_components))
    end_indices = np.concatenate((fd.start_indices[1:], [len(fd.points)]))
    for curve, (start, end) in enumerate(zip(fd.start_indices, end_indices)):
        points = fd.points[start:end, 0]
        observed = fd.values[start:end, 0]
        mean_obs = np.interp(points, GRID, mean)
        components_obs = np.stack(
            [np.interp(points, GRID, component) for component in components],
            axis=1,
        )
        obs_covariance = estimator.covariance_(points, points)
        obs_covariance = (
            obs_covariance + noise * np.eye(len(points))
        )
        centered = observed - mean_obs
        if noise > 0:
            coefficients = np.linalg.solve(obs_covariance, centered)
        else:
            coefficients = np.linalg.lstsq(
                obs_covariance,
                centered,
                rcond=None,
            )[0]
        scores[curve] = eigenvalues * (components_obs.T @ coefficients)
    return scores


class FPCASparseKnownModelTestCase(unittest.TestCase):
    """FPCA with a known estimator: conditional computation checks."""

    def _fit_fpca(
        self,
        n_components: int = 2,
        **kwargs,
    ) -> FPCA:
        fd, _ = _make_curves(30, seed=100)
        fpca = FPCA(
            n_components=n_components,
            covariance_estimator=_KnownEstimator(**kwargs),
        )
        fpca.fit(fd)
        return fpca

    def test_components_are_fdagrid_orthonormal(self) -> None:
        """Components are FDataGrid and orthonormal under integration."""
        fpca = self._fit_fpca()
        self.assertIsInstance(fpca.components_, FDataGrid)
        np.testing.assert_array_equal(
            fpca.components_.grid_points[0],
            GRID,
        )
        components = fpca.components_.data_matrix[..., 0]
        weights = _quadrature_weights()
        gram = components @ (weights[:, None] * components.T)
        np.testing.assert_allclose(
            gram,
            np.eye(fpca.n_components),
            rtol=0,
            atol=1e-12,
        )

    def test_explained_variance(self) -> None:
        """Explained variance matches an independent spectral decomposition."""
        fpca = self._fit_fpca()
        weights = _quadrature_weights()
        sqrt_weights = np.sqrt(weights)
        operator = (
            sqrt_weights[:, None] * _true_kernel(GRID, GRID)
            * sqrt_weights[None, :]
        )
        eigenvalues = np.linalg.eigvalsh(operator)[::-1]

        self.assertTrue(np.all(fpca.explained_variance_ >= 0))
        self.assertTrue(np.all(np.diff(fpca.explained_variance_) <= 0))
        np.testing.assert_allclose(
            fpca.explained_variance_,
            eigenvalues[:2],
            rtol=1e-10,
            atol=1e-12,
        )
        # The discretized eigenvalues are close to the true ones.
        np.testing.assert_allclose(
            fpca.explained_variance_,
            TRUE_EIGENVALUES,
            rtol=1e-6,
        )

    def test_explained_variance_ratio_denominator(self) -> None:
        """The ratio denominator is the whole nonnegative spectrum."""
        fpca_one = self._fit_fpca(n_components=1)
        fpca_two = self._fit_fpca(n_components=2)

        weights = _quadrature_weights()
        sqrt_weights = np.sqrt(weights)
        operator = (
            sqrt_weights[:, None] * _true_kernel(GRID, GRID)
            * sqrt_weights[None, :]
        )
        eigenvalues = np.linalg.eigvalsh(operator)
        total = np.sum(eigenvalues[eigenvalues > 0])

        np.testing.assert_allclose(
            fpca_two.explained_variance_ratio_,
            fpca_two.explained_variance_ / total,
            rtol=1e-12,
        )
        # The denominator does not depend on the number of components.
        self.assertAlmostEqual(
            fpca_one.explained_variance_ratio_[0],
            fpca_two.explained_variance_ratio_[0],
        )
        self.assertAlmostEqual(
            np.sum(fpca_two.explained_variance_ratio_),
            1.0,
        )

    def test_components_match_true_eigenfunctions(self) -> None:
        """Components match the true eigenfunctions up to sign."""
        fpca = self._fit_fpca()
        components = fpca.components_.data_matrix[..., 0]
        true_components = _true_eigenfunctions(GRID).T
        for component, true_component in zip(components, true_components):
            sign = np.sign(np.dot(component, true_component))
            np.testing.assert_allclose(
                sign * component,
                true_component,
                rtol=0,
                atol=1e-5,
            )

    def test_conditional_scores(self) -> None:
        """Scores are the conditional means under the fitted GP model."""
        fpca = self._fit_fpca()
        fd_test, _ = _make_curves(15, seed=200)
        scores = fpca.transform(fd_test)
        np.testing.assert_allclose(
            scores,
            _reference_scores(fpca, fd_test),
            rtol=0,
            atol=1e-10,
        )

    def test_conditional_scores_off_grid_points(self) -> None:
        """Scores at points outside the grid but inside the domain."""
        fpca = self._fit_fpca()
        rng = np.random.default_rng(300)
        points = np.sort(rng.uniform(0.001, 0.999, 6))
        values = _mu(points) + _true_eigenfunctions(points) @ [0.4, -0.3]
        values = values + rng.normal(0.0, 0.1, len(points))
        fd_test = FDataIrregular(
            start_indices=[0],
            points=points[:, np.newaxis],
            values=values[:, np.newaxis],
            domain_range=DOMAIN,
        )
        scores = fpca.transform(fd_test)
        np.testing.assert_allclose(
            scores,
            _reference_scores(fpca, fd_test),
            rtol=0,
            atol=1e-10,
        )

    def test_conditional_scores_fully_staggered(self) -> None:
        """Curves without any common observation time can be scored."""
        fpca = self._fit_fpca()
        rng = np.random.default_rng(400)
        points = []
        values = []
        start_indices = [0]
        for curve in range(6):
            # Disjoint observation windows per curve.
            curve_points = np.sort(
                rng.uniform(curve / 6, (curve + 1) / 6, 4),
            )
            curve_values = (
                _mu(curve_points)
                + _true_eigenfunctions(curve_points) @ [0.1 * curve, -0.2]
            )
            points.append(curve_points)
            values.append(curve_values)
            start_indices.append(start_indices[-1] + 4)
        fd_test = FDataIrregular(
            start_indices=np.array(start_indices[:-1]),
            points=np.concatenate(points)[:, np.newaxis],
            values=np.concatenate(values)[:, np.newaxis],
            domain_range=DOMAIN,
        )
        scores = fpca.transform(fd_test)
        np.testing.assert_allclose(
            scores,
            _reference_scores(fpca, fd_test),
            rtol=0,
            atol=1e-10,
        )

    def test_zero_noise_rank_deficient_legal(self) -> None:
        """Noiseless rank-deficient but consistent observations work."""
        fpca = self._fit_fpca(noise_variance=0.0)
        # Five observations of a rank-2 process: the observation
        # covariance is singular but the observation is in its support.
        points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        obs_covariance = _true_kernel(points, points)
        self.assertLess(np.linalg.matrix_rank(obs_covariance), len(points))

        true_scores = np.array([0.7, -1.1])
        values = _mu(points) + _true_eigenfunctions(points) @ true_scores
        fd_test = FDataIrregular(
            start_indices=[0],
            points=points[:, np.newaxis],
            values=values[:, np.newaxis],
            domain_range=DOMAIN,
        )
        scores = fpca.transform(fd_test)
        self.assertTrue(np.all(np.isfinite(scores)))
        # The conditional scores recover the true latent scores, with
        # the sign convention of the fitted components.
        components = fpca.components_.data_matrix[..., 0]
        true_components = _true_eigenfunctions(GRID).T
        signs = np.sign(np.sum(components * true_components, axis=1))
        np.testing.assert_allclose(
            signs * scores[0],
            true_scores,
            rtol=0,
            atol=1e-8,
        )

    def test_zero_noise_out_of_support_fails(self) -> None:
        """Noiseless observations outside the model support fail."""
        fpca = self._fit_fpca(noise_variance=0.0)
        points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # A high-frequency perturbation cannot be produced by the
        # rank-2 latent process.
        values = _mu(points) + np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        fd_test = FDataIrregular(
            start_indices=[0],
            points=points[:, np.newaxis],
            values=values[:, np.newaxis],
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            fpca.transform(fd_test)

    def test_degenerate_eigenvalues(self) -> None:
        """Degenerate eigenvalues: compare subspaces, not components."""
        fpca = self._fit_fpca(eigenvalues=(0.75, 0.75))
        np.testing.assert_allclose(
            fpca.explained_variance_,
            [0.75, 0.75],
            rtol=1e-6,
        )

        # The estimated principal subspace equals the true one: the
        # projectors P(t, s) = sum_k phi_k(t) phi_k(s) coincide.
        components = fpca.components_.data_matrix[..., 0]
        true_components = _true_eigenfunctions(GRID).T
        projector_estimated = components.T @ components
        projector_true = true_components.T @ true_components
        np.testing.assert_allclose(
            projector_estimated,
            projector_true,
            rtol=0,
            atol=1e-8,
        )

        # Reconstructions are independent of the basis of the
        # degenerate subspace: project a dense noiseless curve.
        latent = _mu(GRID) + _true_eigenfunctions(GRID) @ [0.3, 0.4]
        fd_dense = FDataIrregular(
            start_indices=[0],
            points=GRID[:, np.newaxis],
            values=latent[:, np.newaxis],
            domain_range=DOMAIN,
        )
        fpca_zero = self._fit_fpca(
            eigenvalues=(0.75, 0.75),
            noise_variance=0.0,
        )
        reconstructed = fpca_zero.inverse_transform(
            fpca_zero.transform(fd_dense),
        )
        np.testing.assert_allclose(
            reconstructed.data_matrix[0, :, 0],
            latent,
            rtol=0,
            atol=1e-8,
        )

    def test_inverse_transform(self) -> None:
        """inverse_transform rebuilds latent curves on the model grid."""
        fpca = self._fit_fpca()
        fd_test, _ = _make_curves(5, seed=500)
        scores = fpca.transform(fd_test)
        reconstructed = fpca.inverse_transform(scores)

        self.assertIsInstance(reconstructed, FDataGrid)
        self.assertEqual(
            reconstructed.data_matrix.shape,
            (5, len(GRID), 1),
        )
        np.testing.assert_array_equal(reconstructed.grid_points[0], GRID)

        # Independent reconstruction: mean + sum_k score_k phi_k.
        components = fpca.components_.data_matrix[..., 0]
        mean = fpca.mean_.data_matrix[0, :, 0]
        expected = mean[None, :] + scores @ components
        np.testing.assert_allclose(
            reconstructed.data_matrix[..., 0],
            expected,
            rtol=0,
            atol=1e-12,
        )

    def test_mean_is_estimator_location(self) -> None:
        """The mean used by FPCA is the estimator's location."""
        fpca = self._fit_fpca()
        np.testing.assert_allclose(
            fpca.mean_.data_matrix[0, :, 0],
            _mu(GRID),
        )


class FPCASparseValidationTestCase(unittest.TestCase):
    """Parameter and input validation of the sparse FPCA path."""

    def test_unsupported_parameter_combinations(self) -> None:
        """Non-default dense-path parameters are rejected before fit."""
        fd, _ = _make_curves(5, seed=0)
        with self.assertRaises(NotImplementedError):
            FPCA(
                n_components=2,
                centering=False,
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)
        with self.assertRaises(NotImplementedError):
            FPCA(
                n_components=2,
                regularization=L2Regularization(
                    LinearDifferentialOperator(2),
                ),
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)
        with self.assertRaises(NotImplementedError):
            FPCA(
                n_components=2,
                components_basis=FourierBasis(n_basis=5),
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)
        with self.assertRaises(NotImplementedError):
            FPCA(
                n_components=2,
                _weights=np.ones(len(GRID)),
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)

    def test_n_components_larger_than_grid(self) -> None:
        """More components than grid points are rejected."""
        fd, _ = _make_curves(5, seed=0)
        with self.assertRaises(ValueError):
            FPCA(
                n_components=len(GRID) + 1,
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)

    def test_n_components_larger_than_rank(self) -> None:
        """More components than the operator rank are rejected."""
        fd, _ = _make_curves(5, seed=0)
        with self.assertRaises(ValueError):
            FPCA(
                n_components=3,
                covariance_estimator=_KnownEstimator(),
            ).fit(fd)

    def test_fit_requires_irregular_input(self) -> None:
        """Dense input with a covariance estimator is rejected."""
        fd_grid = FDataGrid(
            np.zeros((3, len(GRID), 1)),
            grid_points=GRID,
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            FPCA(
                n_components=2,
                covariance_estimator=_KnownEstimator(),
            ).fit(fd_grid)

    def test_estimator_protocol_is_checked(self) -> None:
        """An estimator without the required attributes is rejected."""
        fd, _ = _make_curves(5, seed=0)
        with self.assertRaises(TypeError):
            FPCA(
                n_components=2,
                covariance_estimator=_EmptyEstimator(),
            ).fit(fd)

    def test_transform_requires_irregular_input(self) -> None:
        """Dense input cannot be transformed by the sparse FPCA."""
        fd, _ = _make_curves(5, seed=0)
        fpca = FPCA(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        fpca.fit(fd)
        fd_grid = FDataGrid(
            np.zeros((2, len(GRID), 1)),
            grid_points=GRID,
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            fpca.transform(fd_grid)

    def test_transform_rejects_invalid_observations(self) -> None:
        """Non-finite or out-of-domain observations are rejected."""
        fd, _ = _make_curves(5, seed=0)
        fpca = FPCA(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        fpca.fit(fd)

        fd_nan = FDataIrregular(
            start_indices=[0, 2],
            points=[[0.1], [0.2], [0.3], [0.4]],
            values=[[1.0], [np.nan], [1.0], [2.0]],
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            fpca.transform(fd_nan)

        fd_outside = FDataIrregular(
            start_indices=[0, 2],
            points=[[0.1], [1.5], [0.3], [0.4]],
            values=[[1.0], [2.0], [1.0], [2.0]],
            domain_range=DOMAIN,
        )
        with self.assertRaises(ValueError):
            fpca.transform(fd_outside)


class FPCASparseApiTestCase(unittest.TestCase):
    """Estimator API behavior: clone, pickle, immutability, batches."""

    def test_fit_clones_estimator(self) -> None:
        """The caller's estimator is not fitted in place."""
        fd, _ = _make_curves(5, seed=0)
        estimator = _KnownEstimator()
        fpca = FPCA(n_components=2, covariance_estimator=estimator)
        fpca.fit(fd)
        self.assertFalse(hasattr(estimator, "location_"))
        self.assertIsNot(fpca.covariance_estimator_, estimator)
        self.assertIsInstance(
            fpca.covariance_estimator_,
            _KnownEstimator,
        )

    def test_clone_and_get_params(self) -> None:
        """FPCA parameters are managed the sklearn way."""
        estimator = _KnownEstimator()
        fpca = FPCA(n_components=2, covariance_estimator=estimator)
        # sklearn's clone recursively clones the nested estimator.
        cloned = clone(fpca)
        self.assertIsNot(cloned.covariance_estimator, estimator)
        self.assertEqual(
            cloned.covariance_estimator.get_params(),
            estimator.get_params(),
        )
        self.assertEqual(cloned.n_components, 2)
        self.assertIn("covariance_estimator", fpca.get_params())
        # set_params can replace the estimator.
        fpca.set_params(covariance_estimator=_KnownEstimator(noise_variance=0.5))
        self.assertEqual(
            fpca.covariance_estimator.noise_variance,
            0.5,
        )

    def test_fit_does_not_modify_data(self) -> None:
        """The training curves are not mutated by the fit."""
        fd, _ = _make_curves(5, seed=0)
        points_before = fd.points.copy()
        values_before = fd.values.copy()
        fpca = FPCA(n_components=2, covariance_estimator=_KnownEstimator())
        fpca.fit(fd)
        np.testing.assert_array_equal(fd.points, points_before)
        np.testing.assert_array_equal(fd.values, values_before)

    def test_pickle_roundtrip(self) -> None:
        """A fitted FPCA gives the same scores after pickling."""
        fd, _ = _make_curves(10, seed=0)
        fd_test, _ = _make_curves(4, seed=1)
        fpca = FPCA(n_components=2, covariance_estimator=_KnownEstimator())
        fpca.fit(fd)
        restored = pickle.loads(pickle.dumps(fpca))
        np.testing.assert_allclose(
            restored.transform(fd_test),
            fpca.transform(fd_test),
        )

    def test_transform_batch_invariance(self) -> None:
        """Splitting or reordering the batch does not change scores."""
        fd, _ = _make_curves(5, seed=0)
        fd_test, _ = _make_curves(12, seed=1)
        fpca = FPCA(n_components=2, covariance_estimator=_KnownEstimator())
        fpca.fit(fd)

        scores = fpca.transform(fd_test)
        scores_split = np.concatenate(
            [fpca.transform(fd_test[:5]), fpca.transform(fd_test[5:])],
        )
        np.testing.assert_allclose(scores, scores_split)

        permutation = np.random.default_rng(0).permutation(fd_test.n_samples)
        scores_permuted = fpca.transform(fd_test[permutation])
        np.testing.assert_allclose(
            scores,
            scores_permuted[np.argsort(permutation)],
        )

    def test_transform_does_not_update_state(self) -> None:
        """Transforming new curves leaves the fitted state untouched."""
        fd, _ = _make_curves(5, seed=0)
        fd_test, _ = _make_curves(6, seed=1)
        fpca = FPCA(n_components=2, covariance_estimator=_KnownEstimator())
        fpca.fit(fd)
        explained_before = fpca.explained_variance_.copy()
        components_before = fpca.components_.data_matrix.copy()
        mean_before = fpca.mean_.data_matrix.copy()
        scores_first = fpca.transform(fd_test)
        scores_second = fpca.transform(fd_test)
        np.testing.assert_array_equal(scores_first, scores_second)
        np.testing.assert_array_equal(
            fpca.explained_variance_,
            explained_before,
        )
        np.testing.assert_array_equal(
            fpca.components_.data_matrix,
            components_before,
        )
        np.testing.assert_array_equal(
            fpca.mean_.data_matrix,
            mean_before,
        )


class FPCARegressionSparseTestCase(unittest.TestCase):
    """FPCARegression driven by the covariance estimator."""

    def _make_regression_data(
        self,
        n_curves: int,
        seed: int,
    ) -> tuple[FDataIrregular, np.ndarray]:
        """Data from y = 3 + 2 xi_1 - xi_2 + observation noise."""
        fd, true_scores = _make_curves(n_curves, seed=seed)
        rng = np.random.default_rng(seed + 10000)
        y = (
            3.0
            + 2.0 * true_scores[:, 0]
            - 1.0 * true_scores[:, 1]
            + rng.normal(0.0, 0.01, n_curves)
        )
        return fd, y

    def test_regression_recovers_model(self) -> None:
        """The coefficient function and intercept recover the model."""
        fd_train, y_train = self._make_regression_data(300, seed=0)
        regression = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        regression.fit(fd_train, y_train)

        # Coefficients of the linear model in the component basis,
        # with the sign convention of the fitted components.
        coefficients = regression.coef_[0].coefficients.ravel()
        components = regression._fpca.components_.data_matrix[..., 0]
        true_components = _true_eigenfunctions(GRID).T
        signs = np.sign(np.sum(components * true_components, axis=1))
        np.testing.assert_allclose(
            signs * coefficients,
            [2.0, -1.0],
            rtol=0,
            atol=0.1,
        )
        self.assertAlmostEqual(
            regression._linear_model.intercept_[0],
            3.0,
            places=1,
        )

    def test_predictions_use_conditional_scores(self) -> None:
        """Predictions are the linear model of the conditional scores."""
        fd_train, y_train = self._make_regression_data(300, seed=0)
        regression = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        regression.fit(fd_train, y_train)

        fd_test, _ = _make_curves(10, seed=7)
        predictions = regression.predict(fd_test)

        # Independent expected predictions: the fitted linear model
        # applied to independently computed conditional scores.
        scores = _reference_scores(regression._fpca, fd_test)
        coefficients = regression.coef_[0].coefficients.ravel()
        expected = (
            regression._linear_model.intercept_[0]
            + scores @ coefficients
        )
        np.testing.assert_allclose(
            predictions,
            expected,
            rtol=0,
            atol=1e-8,
        )

    def test_model_responds_to_kernel_and_noise(self) -> None:
        """Mean, kernel and noise drive spectrum, scores and regression."""
        fd_train, y_train = self._make_regression_data(100, seed=0)
        fd_test, _ = _make_curves(5, seed=3)

        regression_low = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(noise_variance=0.001),
        )
        regression_low.fit(fd_train, y_train)
        regression_high = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(noise_variance=2.0),
        )
        regression_high.fit(fd_train, y_train)

        # Different noise -> different conditional scores and
        # predictions through the whole pipeline.
        self.assertFalse(
            np.allclose(
                regression_low._fpca.transform(fd_test),
                regression_high._fpca.transform(fd_test),
            ),
        )
        self.assertFalse(
            np.allclose(
                regression_low.predict(fd_test),
                regression_high.predict(fd_test),
            ),
        )

        # Different kernel -> different spectrum, scores and
        # predictions.
        regression_kernel = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(
                eigenvalues=(0.3, 1.2),
            ),
        )
        regression_kernel.fit(fd_train, y_train)
        self.assertFalse(
            np.allclose(
                regression_kernel._fpca.explained_variance_,
                regression_low._fpca.explained_variance_,
            ),
        )
        self.assertFalse(
            np.allclose(
                regression_kernel.predict(fd_test),
                regression_low.predict(fd_test),
            ),
        )

    def test_regression_regularization_still_applies(self) -> None:
        """The regression regularization changes the fitted model."""
        fd_train, y_train = self._make_regression_data(100, seed=0)
        regression_plain = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        regression_plain.fit(fd_train, y_train)
        regression_penalized = FPCARegression(
            n_components=2,
            regression_regularization=L2Regularization(
                LinearDifferentialOperator(2),
                regularization_parameter=1.0,
            ),
            covariance_estimator=_KnownEstimator(),
        )
        regression_penalized.fit(fd_train, y_train)
        self.assertFalse(
            np.allclose(
                regression_plain.coef_[0].coefficients,
                regression_penalized.coef_[0].coefficients,
            ),
        )

    def test_regression_pickle_roundtrip(self) -> None:
        """A fitted regression gives the same predictions after pickling."""
        fd_train, y_train = self._make_regression_data(50, seed=0)
        fd_test, _ = _make_curves(4, seed=9)
        regression = FPCARegression(
            n_components=2,
            covariance_estimator=_KnownEstimator(),
        )
        regression.fit(fd_train, y_train)
        restored = pickle.loads(pickle.dumps(regression))
        np.testing.assert_allclose(
            restored.predict(fd_test),
            regression.predict(fd_test),
        )


class FPCASparseEndToEndTestCase(unittest.TestCase):
    """Sparse estimation + FPCA + reconstruction on a known process.

    The true process is smooth and low-rank (three sinusoidal
    eigenfunctions) with independent noise of variance 0.04, observed
    at 2 to 8 random times per curve. The reconstruction of held-out
    curves is evaluated against their true dense latent paths, which
    were not used in the fit. The baseline interpolates each curve
    linearly onto the grid and applies the classical (dense) FPCA.

    Tolerances: the reconstruction error of the sparse path with a
    handful of noisy observations per curve is bounded well above its
    observed range (0.25-0.35 over the seeds); the comparison with the
    baseline is done on medians over seeds, since a single seed can be
    unfavorable to either method.
    """

    n_components = 3
    seeds = (0, 1, 2, 3, 4)

    @staticmethod
    def _to_grid(fd: FDataIrregular) -> FDataGrid:
        """Per-curve linear interpolation onto the grid (baseline)."""
        data_matrix = np.empty((fd.n_samples, len(GRID)))
        end_indices = np.concatenate((fd.start_indices[1:], [len(fd.points)]))
        for curve, (start, end) in enumerate(zip(fd.start_indices, end_indices)):
            data_matrix[curve] = np.interp(
                GRID,
                fd.points[start:end, 0],
                fd.values[start:end, 0],
            )
        return FDataGrid(
            data_matrix[:, :, np.newaxis],
            grid_points=GRID,
            domain_range=DOMAIN,
        )

    def _reconstruction_errors(self, seed: int) -> tuple[float, float]:
        """Reconstruction errors of the sparse and baseline paths."""
        train = _make_sparse_sample_lowrank(1000, seed=5000 + seed)
        test, latent_true = _make_dense_test_curves(100, seed=9000 + seed)

        fpca = FPCA(
            n_components=self.n_components,
            covariance_estimator=EmpiricalCovariance(
                sparse=True,
                grid_points=GRID,
            ),
        )
        fpca.fit(train)
        reconstructed = fpca.inverse_transform(fpca.transform(test))
        sparse_error = _relative_l2(
            reconstructed.data_matrix[..., 0],
            latent_true,
        )

        baseline_fpca = FPCA(n_components=self.n_components)
        baseline_fpca.fit(self._to_grid(train))
        baseline_reconstructed = baseline_fpca.inverse_transform(
            baseline_fpca.transform(self._to_grid(test)),
        )
        baseline_error = _relative_l2(
            baseline_reconstructed.data_matrix[..., 0],
            latent_true,
        )
        return float(np.mean(sparse_error)), float(np.mean(baseline_error))

    def test_reconstruction_beats_interpolation_baseline(self) -> None:
        """The sparse path reconstructs held-out latent curves better."""
        sparse_errors = []
        baseline_errors = []
        for seed in self.seeds:
            sparse_error, baseline_error = self._reconstruction_errors(seed)
            sparse_errors.append(sparse_error)
            baseline_errors.append(baseline_error)
            # Bounded in absolute terms, with margin over the observed
            # range (see the class docstring).
            self.assertLess(sparse_error, 0.5)

        self.assertLess(
            float(np.median(sparse_errors)),
            float(np.median(baseline_errors)),
        )
        # The sparse path wins in at least most of the seeds.
        wins = sum(
            sparse < baseline
            for sparse, baseline in zip(sparse_errors, baseline_errors)
        )
        self.assertGreaterEqual(wins, len(self.seeds) - 1)


# True low-rank model of the end-to-end tests: three sinusoidal
# eigenfunctions with eigenvalues (1, 0.5, 0.25) and independent noise
# of variance 0.04.
LOWRANK_EIGENVALUES = np.array([1.0, 0.5, 0.25])
LOWRANK_NOISE = 0.04


def _lowrank_mean(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * t


def _lowrank_eigenfunctions(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.stack(
        [
            np.sqrt(2.0) * np.sin(2 * np.pi * t),
            np.sqrt(2.0) * np.cos(2 * np.pi * t),
            np.sqrt(2.0) * np.sin(4 * np.pi * t),
        ],
        axis=-1,
    )


def _make_sparse_sample_lowrank(n_curves: int, seed: int) -> FDataIrregular:
    """Sparse noisy observations of the low-rank process."""
    rng = np.random.default_rng(seed)
    points = []
    values = []
    start_indices = [0]
    for _ in range(n_curves):
        n_points = rng.integers(2, 9)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        scores = rng.normal(0.0, np.sqrt(LOWRANK_EIGENVALUES))
        latent = (
            _lowrank_mean(curve_points)
            + _lowrank_eigenfunctions(curve_points) @ scores
        )
        observed = latent + rng.normal(0.0, np.sqrt(LOWRANK_NOISE), n_points)
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)

    return FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=DOMAIN,
    )


def _make_dense_test_curves(
    n_curves: int,
    seed: int,
) -> tuple[FDataIrregular, np.ndarray]:
    """Dense latent curves (truth) and their sparse noisy observations."""
    rng = np.random.default_rng(seed)
    latent = []
    points = []
    values = []
    start_indices = [0]
    for _ in range(n_curves):
        scores = rng.normal(0.0, np.sqrt(LOWRANK_EIGENVALUES))
        latent.append(
            _lowrank_mean(GRID) + _lowrank_eigenfunctions(GRID) @ scores,
        )

        n_points = rng.integers(2, 9)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        observed = (
            _lowrank_mean(curve_points)
            + _lowrank_eigenfunctions(curve_points) @ scores
            + rng.normal(0.0, np.sqrt(LOWRANK_NOISE), n_points)
        )
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)

    fd = FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=DOMAIN,
    )
    return fd, np.array(latent)


def _relative_l2(reconstructed: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Relative L2 reconstruction error of each curve on the grid."""
    numerator = scipy.integrate.simpson((reconstructed - true) ** 2, x=GRID, axis=1)
    denominator = scipy.integrate.simpson(true ** 2, x=GRID, axis=1)
    return np.sqrt(numerator / denominator)


if __name__ == "__main__":
    unittest.main()
