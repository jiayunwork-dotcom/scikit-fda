"""
Sparse FPCA: principal components from irregular noisy observations
===================================================================

Analyzing curves that are observed only at a few, subject-specific and
noisy times, without interpolating them first: sparse covariance
estimation, conditional FPCA scores and FPCA regression, including how
the estimators fail when their assumptions do not hold.
"""

# Author: scikit-fda developers
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from skfda.exploratory.stats.covariance import (
    CovarianceEstimator,
    EmpiricalCovariance,
)
from skfda.misc.covariances import EigenExpansion
from skfda.ml.regression import FPCARegression
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation import FDataGrid, FDataIrregular

##############################################################################
# In many longitudinal studies each subject is measured only a handful of
# times, at different and irregularly spaced times, and the measurements
# are contaminated by noise. Interpolating each subject to a dense grid
# before the analysis treats the unobserved stretches of the curves as if
# they were observed, and confounds the latent process variation with the
# measurement noise. This example shows, on simulated data with a known
# ground truth, how the sparse analysis path avoids both problems, and
# what happens when its assumptions are violated.
#
# The true process is a smooth low-rank process with three sinusoidal
# eigenfunctions, observed with independent noise. Each curve is observed
# at 2 to 8 random times.
grid = np.linspace(0.0, 1.0, 81)
domain = [(0.0, 1.0)]
eigenvalues_true = np.array([1.0, 0.5, 0.25])
noise_true = 0.04


def mean_true(t):
    return np.sin(2 * np.pi * t) + 0.5 * t


def eigenfunctions_true(t):
    t = np.asarray(t, dtype=float)
    return np.stack(
        [
            np.sqrt(2.0) * np.sin(2 * np.pi * t),
            np.sqrt(2.0) * np.cos(2 * np.pi * t),
            np.sqrt(2.0) * np.sin(4 * np.pi * t),
        ],
        axis=-1,
    )


def make_sample(n_curves, seed):
    rng = np.random.default_rng(seed)
    points, values, start_indices = [], [], [0]
    scores_all = []
    for _ in range(n_curves):
        n_points = rng.integers(2, 9)
        curve_points = np.sort(rng.uniform(0.0, 1.0, n_points))
        scores = rng.normal(0.0, np.sqrt(eigenvalues_true))
        latent = (
            mean_true(curve_points)
            + eigenfunctions_true(curve_points) @ scores
        )
        observed = latent + rng.normal(0.0, np.sqrt(noise_true), n_points)
        points.append(curve_points)
        values.append(observed)
        start_indices.append(start_indices[-1] + n_points)
        scores_all.append(scores)
    fd = FDataIrregular(
        start_indices=np.array(start_indices[:-1]),
        points=np.concatenate(points)[:, np.newaxis],
        values=np.concatenate(values)[:, np.newaxis],
        domain_range=domain,
    )
    return fd, np.array(scores_all)


fd_train, _ = make_sample(2000, seed=0)
fd_train.plot()
plt.show()

##############################################################################
# Sparse covariance estimation
# ----------------------------
#
# The sparse mode of :class:`~skfda.exploratory.stats.covariance.EmpiricalCovariance`
# pools the observations of all curves to smooth the mean, and pools the
# within-curve cross-products of the residuals (excluding the diagonal,
# which is contaminated by the noise) to smooth the covariance surface.
# The noise variance is separated from the latent process variance, and
# the returned kernel is positive semi-definite everywhere on the domain.
estimator = EmpiricalCovariance(sparse=True, grid_points=grid)
estimator.fit(fd_train)

covariance_true = sum(
    lam * np.outer(phi, phi)
    for lam, phi in zip(eigenvalues_true, eigenfunctions_true(grid).T)
)
covariance_estimated = estimator.covariance_(grid, grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, surface, title in zip(
    axes,
    (covariance_true, covariance_estimated),
    ("True covariance", "Estimated covariance"),
):
    image = ax.imshow(
        surface,
        extent=[0, 1, 1, 0],
        vmin=-covariance_true.max(),
        vmax=covariance_true.max(),
        cmap="RdBu_r",
    )
    ax.set_title(title)
fig.colorbar(image, ax=axes)
plt.show()

##############################################################################
# The estimation quality is summarized by the noise variance estimate,
# the integrated covariance error and the error in the principal
# subspace, all computable here because the ground truth is known.
weights = scipy.integrate.simpson(np.eye(len(grid)), x=grid)


def subspace_error(estimated, true, n_components=3):
    sqrt_weights = np.sqrt(weights)

    def top_vectors(cov):
        _, vectors = np.linalg.eigh(
            sqrt_weights[:, None] * cov * sqrt_weights[None, :],
        )
        return vectors[:, ::-1][:, :n_components]

    singular = np.linalg.svd(
        top_vectors(estimated).T @ top_vectors(true),
        compute_uv=False,
    )
    return np.sqrt(max(0.0, 1.0 - singular[-1] ** 2))


covariance_error = np.sqrt(
    scipy.integrate.simpson(
        scipy.integrate.simpson(
            (covariance_estimated - covariance_true) ** 2, x=grid, axis=1,
        ),
        x=grid,
    )
    / scipy.integrate.simpson(
        scipy.integrate.simpson(covariance_true ** 2, x=grid, axis=1),
        x=grid,
    ),
)
print(f"Noise variance: estimated {estimator.noise_variance_:.4f}, "
      f"true {noise_true:.4f}")
print(f"Covariance integrated relative error: {covariance_error:.3f}")
print(f"Principal subspace error: "
      f"{subspace_error(covariance_estimated, covariance_true):.3f}")

##############################################################################
# Conditional FPCA scores
# -----------------------
#
# Passing the estimator to :class:`~skfda.preprocessing.dim_reduction.FPCA`
# gives the principal components of the estimated covariance operator.
# The scores of a new curve are then the means of its principal component
# scores conditioned on its few noisy observations, under the fitted
# Gaussian process model: no interpolation of the new curve is involved.
fpca = FPCA(n_components=3, covariance_estimator=estimator)
fpca.fit(fd_train)

fd_test, _ = make_sample(6, seed=1)
scores = fpca.transform(fd_test)
reconstructed = fpca.inverse_transform(scores)
reconstructed.plot()
plt.show()

##############################################################################
# FPCA regression
# ---------------
#
# The same conditional scores drive :class:`~skfda.ml.regression.FPCARegression`:
# both the training design matrix and the predictions are linear in the
# conditional scores, so the regression works directly on the sparse
# observations.
fd_reg_train, scores_reg = make_sample(300, seed=2)
y_train = 3.0 + 2.0 * scores_reg[:, 0] - scores_reg[:, 1]
regression = FPCARegression(n_components=3, covariance_estimator=estimator)
regression.fit(fd_reg_train, y_train)

fd_reg_test, scores_reg_test = make_sample(50, seed=3)
y_test = 3.0 + 2.0 * scores_reg_test[:, 0] - scores_reg_test[:, 1]
print(f"Regression R^2 on held-out curves: "
      f"{regression.score(fd_reg_test, y_test):.3f}")

##############################################################################
# When the assumptions do not hold
# --------------------------------
#
# The estimator refuses clearly invalid input instead of returning a
# degenerate answer. For example, a curve with a single observation does
# not carry any within-curve covariance information:
fd_one_point = FDataIrregular(
    start_indices=[0, 2, 3, 5],
    points=[[0.1], [0.2], [0.3], [0.1], [0.2]],
    values=[[1.0], [2.0], [3.0], [1.0], [2.0]],
    domain_range=domain,
)
try:
    EmpiricalCovariance(sparse=True, grid_points=grid).fit(fd_one_point)
except ValueError as error:
    print(f"Single-observation curve -> {error}")

##############################################################################
# When the observations do not cover the domain
# ---------------------------------------------
#
# If the observations leave a region of the domain unsupported, the
# smoothing system is singular and the fit fails explicitly (with the
# default small penalty the fit completes, but the unsupported region is
# driven by the penalty rather than by the data).
rng = np.random.default_rng(0)
points = [np.sort(rng.uniform(0.0, 0.3, 5)) for _ in range(30)]
fd_hole = FDataIrregular(
    start_indices=5 * np.arange(30),
    points=np.concatenate(points)[:, np.newaxis],
    values=np.sin(np.concatenate(points))[:, np.newaxis],
    domain_range=domain,
)
try:
    EmpiricalCovariance(
        sparse=True,
        grid_points=grid,
        smoothing_parameter=0.0,
    ).fit(fd_hole)
except ValueError as error:
    print(f"Unsupported region -> {error}")

##############################################################################
# Numerical failure: observations incompatible with a noiseless model
# -------------------------------------------------------------------
#
# With zero noise and an exactly low-rank kernel, the observation
# covariance is singular. Observations inside the model support still
# get finite scores, but observations that no latent curve of the model
# can produce are rejected instead of being accommodated by silently
# perturbing the model. To show it we use a small didactic estimator
# that returns the true rank-2 kernel; it also illustrates the protocol
# (location_, covariance_, noise_variance_) that a custom estimator
# must provide to work with FPCA.


class LowRankEstimator(CovarianceEstimator):
    """Didactic estimator returning a known rank-2 noiseless model."""

    def fit(self, X, y=None):
        self.location_ = FDataGrid(
            mean_true(grid)[np.newaxis, :, np.newaxis],
            grid_points=grid,
            domain_range=domain,
        )
        self.covariance_ = EigenExpansion(
            grid_points=grid,
            eigenvalues=eigenvalues_true[:2],
            eigenvectors=eigenfunctions_true(grid)[:, :2],
        )
        self.noise_variance_ = 0.0
        return self


fpca_zero = FPCA(n_components=2, covariance_estimator=LowRankEstimator())
fpca_zero.fit(fd_train)

# Observations from a true latent curve of the model: the observation
# covariance is singular (rank 2 for 5 points), but the scores are
# still computed, because the observations lie in the model support.
points_ok = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
values_ok = (
    mean_true(points_ok)
    + eigenfunctions_true(points_ok)[:, :2] @ [0.7, -1.1]
)
fd_ok = FDataIrregular(
    start_indices=[0],
    points=points_ok[:, np.newaxis],
    values=values_ok[:, np.newaxis],
    domain_range=domain,
)
print(f"Scores of in-support observations: {fpca_zero.transform(fd_ok)}")

# A high-frequency perturbation cannot be produced by any latent curve
# of the rank-2 noiseless model, so it is rejected.
values_bad = mean_true(points_ok) + np.array([0.0, 1.0, 0.0, -1.0, 0.0])
fd_bad = FDataIrregular(
    start_indices=[0],
    points=points_ok[:, np.newaxis],
    values=values_bad[:, np.newaxis],
    domain_range=domain,
)
try:
    fpca_zero.transform(fd_bad)
except ValueError as error:
    print(f"Out-of-support observation -> {error}")
