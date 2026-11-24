"""Sparse functional covariance estimation.

This module implements the estimation of the mean function, the covariance
kernel and the measurement noise variance of a stochastic process observed
through sparse and noisy measurements, as described in the documentation of
:class:`skfda.exploratory.stats.covariance.EmpiricalCovariance`.

The approach follows the PACE methodology (Yao, Müller and Wang, 2005,
"Functional Data Analysis for Sparse Longitudinal Data", JASA 100: 577-590),
but uses global penalized B-spline smoothers instead of local polynomial
smoothers:

* The mean function is estimated by a penalized cubic B-spline regression
  of all the pooled observations.
* The covariance surface is estimated by a penalized tensor-product
  B-spline regression of the pooled within-curve cross-products of the
  residuals, excluding the diagonal (which is contaminated by the
  measurement noise). Only products of observations of the *same* curve
  are used, so the estimate exploits the within-curve covariance
  information without assuming common observation times across curves.
* The measurement noise variance is estimated as the difference between a
  penalized spline smooth of the squared residuals (which contains the
  process variance plus the noise variance) and the diagonal of the
  smoothed covariance surface (which contains only the process variance),
  averaged over the domain.
* The smoothed covariance surface is projected onto the cone of positive
  semi-definite matrices by an eigen-decomposition on the evaluation grid,
  and the kernel is defined on the whole domain as the corresponding
  eigen-expansion with linearly interpolated eigenfunctions. This makes
  the resulting kernel positive semi-definite on every finite set of
  domain points, not only on the training grid.

"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.integrate
import scipy.interpolate

from ....misc.covariances import EigenExpansion
from ....representation import FDataGrid, FDataIrregular
from ....typing._numpy import ArrayLike, NDArrayFloat, NDArrayInt

_BSPLINE_DEGREE = 3

# Fraction of the domain trimmed at each boundary when averaging the
# diagonal difference in the noise variance estimation. The diagonal of
# the smoothed covariance surface is less reliable near the boundaries of
# the domain, where the within-curve pairs are scarce.
_NOISE_TRIM_FRACTION = 0.1


class SparseData(NamedTuple):
    """Packed sparse observations of a sample of curves."""

    points: NDArrayFloat
    values: NDArrayFloat
    start_indices: NDArrayInt
    n_samples: int
    domain_range: tuple[float, float]


def _validate_grid_points(
    grid_points: ArrayLike | None,
    domain_range: tuple[float, float],
) -> NDArrayFloat:
    """Validate the evaluation grid provided by the caller."""
    if grid_points is None:
        raise ValueError(
            "grid_points must be provided when sparse=True: it defines "
            "the grid on which the mean and the covariance operator are "
            "estimated.",
        )

    grid = np.asarray(grid_points, dtype=np.float64)

    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError(
            "grid_points must be a one-dimensional array with at least "
            "3 points.",
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid_points must contain only finite values.")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("grid_points must be strictly increasing.")

    domain_start, domain_end = domain_range
    if grid[0] != domain_start or grid[-1] != domain_end:
        raise ValueError(
            "The first and last points of grid_points must be equal to "
            "the extremes of the domain range of the data "
            f"({domain_start}, {domain_end}), but they are "
            f"({grid[0]}, {grid[-1]}).",
        )

    return grid


def _validate_noise_variance(noise_variance: object) -> None:
    """Validate the noise_variance parameter."""
    if isinstance(noise_variance, str):
        if noise_variance != "auto":
            raise ValueError(
                'noise_variance must be "auto" or a finite nonnegative '
                f"real number, but it is {noise_variance!r}.",
            )
        return

    if isinstance(noise_variance, bool):
        raise ValueError(
            'noise_variance must be "auto" or a finite nonnegative real '
            f"number, but it is the boolean {noise_variance!r}.",
        )

    try:
        noise_value = float(noise_variance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            'noise_variance must be "auto" or a finite nonnegative '
            f"real number, but it is {noise_variance!r}.",
        ) from None

    if not np.isfinite(noise_value) or noise_value < 0:
        raise ValueError(
            "noise_variance must be a finite nonnegative real number, "
            f"but it is {noise_value}.",
        )


def _pack_sparse_data(X: FDataIrregular) -> SparseData:
    """Validate the sparse input and pack its observations.

    Raises:
        NotImplementedError: If the domain or codomain dimensions are not
            one.
        ValueError: If the sample does not satisfy the requirements of the
            sparse estimator.

    Returns:
        The packed observations.

    """
    if X.dim_domain != 1 or X.dim_codomain != 1:
        raise NotImplementedError(
            "Sparse covariance estimation is only implemented for "
            "functions with one-dimensional domain and codomain, but "
            f"dim_domain={X.dim_domain} and dim_codomain="
            f"{X.dim_codomain}.",
        )

    n_samples = X.n_samples
    if n_samples < 3:
        raise ValueError(
            "Sparse covariance estimation requires at least 3 curves, "
            f"but only {n_samples} were provided.",
        )

    points = np.asarray(X.points[:, 0], dtype=np.float64)
    values = np.asarray(X.values[:, 0], dtype=np.float64)
    start_indices = np.asarray(X.start_indices)

    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Sparse covariance estimation requires all observation "
            "points and values to be finite.",
        )

    domain_range, = X.domain_range
    domain_start, domain_end = float(domain_range[0]), float(domain_range[1])
    if np.any(points < domain_start) or np.any(points > domain_end):
        raise ValueError(
            "All observation points must lie inside the declared domain "
            f"range ({domain_start}, {domain_end}).",
        )

    end_indices = np.concatenate((start_indices[1:], [len(points)]))
    n_points_per_curve = end_indices - start_indices
    if np.any(n_points_per_curve < 2):
        raise ValueError(
            "Sparse covariance estimation requires at least 2 "
            "observations per curve.",
        )

    for start, end in zip(start_indices, end_indices):
        curve_points = points[start:end]
        if not np.all(np.diff(curve_points) > 0):
            raise ValueError(
                "The observation points of each curve must be strictly "
                "increasing (duplicated time points within a curve are "
                "not allowed).",
            )

    return SparseData(
        points=points,
        values=values,
        start_indices=start_indices,
        n_samples=n_samples,
        domain_range=(domain_start, domain_end),
    )


def _bspline_knots(
    n_basis: int,
    domain_range: tuple[float, float],
) -> NDArrayFloat:
    """Compute the knots of a clamped B-spline basis on the domain."""
    n_interior = n_basis - _BSPLINE_DEGREE - 1
    domain_start, domain_end = domain_range
    interior = np.linspace(
        domain_start,
        domain_end,
        num=n_interior + 2,
    )[1:-1]
    boundary = np.linspace(domain_start, domain_end, num=2)
    return np.concatenate(
        (
            np.repeat(boundary[0], _BSPLINE_DEGREE + 1),
            interior,
            np.repeat(boundary[1], _BSPLINE_DEGREE + 1),
        ),
    )


def _bspline_design(
    points: NDArrayFloat,
    knots: NDArrayFloat,
) -> NDArrayFloat:
    """Evaluate the B-spline design matrix at the given points."""
    design = scipy.interpolate.BSpline.design_matrix(
        points,
        knots,
        _BSPLINE_DEGREE,
        extrapolate=True,
    )
    return np.asarray(design.todense())


def _difference_penalty(n_basis: int) -> NDArrayFloat:
    """Second-order difference penalty matrix of a B-spline basis."""
    second_difference = np.diff(np.eye(n_basis), n=2, axis=0)
    return second_difference.T @ second_difference


def _solve_penalized_normal_equations(
    design: NDArrayFloat,
    rhs: NDArrayFloat,
    penalty: NDArrayFloat,
    smoothing_parameter: float,
    *,
    context: str,
) -> NDArrayFloat:
    """Solve (design' design + lambda P) x = design' rhs.

    The penalty is scaled so that ``smoothing_parameter`` is roughly
    independent of the number of observations and of the basis size: the
    multiplier is ``smoothing_parameter * trace(design' design) /
    trace(penalty)``.

    Raises:
        ValueError: If the system is numerically singular.

    """
    gram = design.T @ design
    rhs_transformed = design.T @ rhs

    if smoothing_parameter > 0:
        penalty_trace = np.trace(penalty)
        gram_trace = np.trace(gram)
        if penalty_trace > 0 and gram_trace > 0:
            gram = gram + (
                smoothing_parameter * gram_trace / penalty_trace
            ) * penalty

    try:
        return np.linalg.solve(gram, rhs_transformed)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"The {context} smoother is numerically singular: the "
            "observations do not provide enough support for the "
            "requested basis. Provide more observations, reduce the "
            "number of basis functions or increase the smoothing "
            "parameter.",
        ) from error


def _fit_mean(
    data: SparseData,
    grid: NDArrayFloat,
    n_basis: int,
    smoothing_parameter: float,
) -> NDArrayFloat:
    """Fit the penalized spline smoother of the mean function.

    Returns:
        The estimated mean evaluated at the grid points.

    """
    knots = _bspline_knots(n_basis, data.domain_range)
    design = _bspline_design(data.points, knots)
    penalty = _difference_penalty(n_basis)
    coefficients = _solve_penalized_normal_equations(
        design,
        data.values,
        penalty,
        smoothing_parameter,
        context="mean",
    )
    grid_design = _bspline_design(grid, knots)
    return grid_design @ coefficients


def _within_curve_pairs(
    data: SparseData,
    residuals: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute the off-diagonal within-curve cross-products.

    Only pairs of observations of the same curve are used. Both orders
    of each pair are included, so the resulting surface estimate is
    symmetric by construction.

    Returns:
        Arrays with the first argument, the second argument and the
        cross-product of the residuals of every ordered off-diagonal
        pair.

    """
    end_indices = np.concatenate(
        (data.start_indices[1:], [len(data.points)]),
    )

    first_indices = []
    second_indices = []
    for start, end in zip(data.start_indices, end_indices):
        n_points = end - start
        curve_indices = np.arange(start, end)
        row_indices = np.repeat(curve_indices, n_points - 1)
        # For each observation, the indices of the other observations
        # of the same curve.
        col_indices = np.concatenate(
            [
                np.delete(curve_indices, position)
                for position in range(n_points)
            ],
        )
        first_indices.append(row_indices)
        second_indices.append(col_indices)

    first = np.concatenate(first_indices)
    second = np.concatenate(second_indices)

    return (
        data.points[first],
        data.points[second],
        residuals[first] * residuals[second],
    )


def _fit_covariance_surface(
    data: SparseData,
    residuals: NDArrayFloat,
    grid: NDArrayFloat,
    n_basis: int,
    smoothing_parameter: float,
) -> NDArrayFloat:
    """Fit the penalized tensor spline of the covariance surface.

    Returns:
        The covariance matrix evaluated at the cartesian product of the
        grid with itself, before the positive semi-definite projection.

    """
    pair_first, pair_second, pair_products = _within_curve_pairs(
        data,
        residuals,
    )

    knots = _bspline_knots(n_basis, data.domain_range)
    penalty_1d = _difference_penalty(n_basis)
    identity = np.eye(n_basis)
    penalty_2d = np.kron(penalty_1d, identity) + np.kron(
        identity,
        penalty_1d,
    )

    # The design matrix of the tensor-product fit is the row-wise
    # Kronecker product of the marginal design matrices. Its normal
    # equations are accumulated in chunks to bound the memory usage:
    # only O(chunk_size * n_basis^2) doubles are alive at once,
    # independently of the number of observations.
    n_basis_squared = n_basis * n_basis
    gram = np.zeros((n_basis_squared, n_basis_squared))
    rhs = np.zeros((n_basis, n_basis))

    chunk_size = 16384
    for chunk_start in range(0, len(pair_products), chunk_size):
        chunk = slice(chunk_start, chunk_start + chunk_size)
        design_first = _bspline_design(pair_first[chunk], knots)
        design_second = _bspline_design(pair_second[chunk], knots)

        # gram_chunk[(i, k), (j, l)] =
        #     sum_p B_i(s_p) B_k(s_p) B_j(t_p) B_l(t_p)
        gram_first = (
            design_first[:, :, np.newaxis] * design_first[:, np.newaxis, :]
        ).reshape(-1, n_basis_squared)
        gram_second = (
            design_second[:, :, np.newaxis] * design_second[:, np.newaxis, :]
        ).reshape(-1, n_basis_squared)
        gram_chunk = gram_first.T @ gram_second

        # Rearrange the (i, k), (j, l) axes as (i, j), (k, l) so that the
        # row and column indices match the row-wise Kronecker product
        # design matrix B(s) (x) B(t).
        gram += gram_chunk.reshape(
            n_basis, n_basis, n_basis, n_basis,
        ).transpose(0, 2, 1, 3).reshape(n_basis_squared, n_basis_squared)

        # rhs[(i, j)] = sum_p B_i(s_p) B_j(t_p) g_p
        rhs += (
            design_first * pair_products[chunk, np.newaxis]
        ).T @ design_second

    if smoothing_parameter > 0:
        penalty_trace = np.trace(penalty_2d)
        gram_trace = np.trace(gram)
        if penalty_trace > 0 and gram_trace > 0:
            gram = gram + (
                smoothing_parameter * gram_trace / penalty_trace
            ) * penalty_2d

    try:
        coefficients = np.linalg.solve(gram, rhs.ravel())
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The covariance surface smoother is numerically singular: "
            "the within-curve pairs do not provide enough support for "
            "the requested basis. Provide more observations per curve, "
            "reduce cov_n_basis or increase the smoothing parameter.",
        ) from error

    coefficient_matrix = coefficients.reshape(n_basis, n_basis)
    grid_design = _bspline_design(grid, knots)
    covariance_grid = grid_design @ coefficient_matrix @ grid_design.T

    # Remove the asymmetry introduced by floating point roundoff.
    return (covariance_grid + covariance_grid.T) / 2


def _project_to_psd(
    covariance_grid: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Project a symmetric matrix onto the positive semi-definite cone.

    Returns:
        The nonnegative eigenvalues and the corresponding eigenvectors
        of the projection.

    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_grid)
    nonnegative = eigenvalues > 0
    return eigenvalues[nonnegative], eigenvectors[:, nonnegative]


def _estimate_noise_variance(
    data: SparseData,
    residuals: NDArrayFloat,
    covariance_grid: NDArrayFloat,
    grid: NDArrayFloat,
    n_basis: int,
    smoothing_parameter: float,
) -> float:
    """Estimate the measurement noise variance.

    The diagonal of the raw covariance of the observations contains the
    process variance plus the noise variance, while the smoothed
    covariance surface (fitted without the diagonal pairs) contains only
    the process variance. Their difference, averaged over the domain, is
    the estimated noise variance.

    The average is computed over the interior of the domain, trimming a
    fraction ``_NOISE_TRIM_FRACTION`` at each boundary, because the
    diagonal of the smoothed surface is less reliable near the
    boundaries, where the within-curve pairs are scarce. If the grid has
    too few interior points, the whole domain is used instead.
    """
    knots = _bspline_knots(n_basis, data.domain_range)
    design = _bspline_design(data.points, knots)
    penalty = _difference_penalty(n_basis)
    squared_residuals = residuals ** 2
    coefficients = _solve_penalized_normal_equations(
        design,
        squared_residuals,
        penalty,
        smoothing_parameter,
        context="noise variance",
    )
    grid_design = _bspline_design(grid, knots)
    diagonal_with_noise = grid_design @ coefficients
    diagonal_process = np.diag(covariance_grid)
    difference = diagonal_with_noise - diagonal_process

    domain_start, domain_end = data.domain_range
    trim = _NOISE_TRIM_FRACTION * (domain_end - domain_start)
    interior = (grid >= domain_start + trim) & (grid <= domain_end - trim)
    if np.sum(interior) < 3:
        interior = np.ones_like(grid, dtype=bool)
    interior_grid = grid[interior]

    noise = scipy.integrate.simpson(
        difference[interior],
        x=interior_grid,
    ) / (interior_grid[-1] - interior_grid[0])

    # A negative estimate corresponds to a noise variance of zero: the
    # smoothed diagonal already lies below the process variance.
    return float(max(noise, 0.0))


def fit_sparse_covariance(
    X: FDataIrregular,
    *,
    grid_points: ArrayLike | None,
    noise_variance: float | str,
    assume_centered: bool,
    mean_n_basis: int,
    cov_n_basis: int,
    smoothing_parameter: float,
) -> tuple[FDataGrid, EigenExpansion, float]:
    """Estimate mean, covariance kernel and noise from sparse data.

    Args:
        X: Sparse functional data.
        grid_points: Evaluation grid of the mean and of the covariance
            operator. Its extremes must coincide with the domain range.
        noise_variance: ``"auto"`` to estimate the noise variance from
            the data, or a fixed nonnegative value.
        assume_centered: If ``True``, the mean is assumed to be zero.
        mean_n_basis: Number of B-spline basis functions of the mean
            smoother.
        cov_n_basis: Number of B-spline basis functions per dimension of
            the covariance surface smoother.
        smoothing_parameter: Relative penalty of the smoothers.

    Returns:
        The estimated mean as a one-sample FDataGrid on the evaluation
        grid, the estimated process covariance kernel (without
        measurement noise) and the estimated noise variance.

    """
    if not isinstance(X, FDataIrregular):
        raise ValueError(
            "Sparse covariance estimation requires a FDataIrregular "
            f"object, but {type(X).__name__} was provided.",
        )

    _validate_noise_variance(noise_variance)

    for name, n_basis in (
        ("mean_n_basis", mean_n_basis),
        ("cov_n_basis", cov_n_basis),
    ):
        if isinstance(n_basis, bool) or not np.issubdtype(
            np.asarray(n_basis).dtype,
            np.integer,
        ):
            raise ValueError(f"{name} must be an integer.")
        if n_basis < _BSPLINE_DEGREE + 1:
            raise ValueError(
                f"{name} must be at least {_BSPLINE_DEGREE + 1}, but "
                f"it is {n_basis}.",
            )

    if not np.isfinite(smoothing_parameter) or smoothing_parameter < 0:
        raise ValueError(
            "smoothing_parameter must be a finite nonnegative real "
            f"number, but it is {smoothing_parameter}.",
        )

    data = _pack_sparse_data(X)
    grid = _validate_grid_points(grid_points, data.domain_range)

    mean_grid = _fit_mean(
        data,
        grid,
        mean_n_basis,
        smoothing_parameter,
    )
    if assume_centered:
        mean_grid = np.zeros_like(mean_grid)

    residuals = data.values - np.interp(
        data.points,
        grid,
        mean_grid,
    )

    covariance_grid = _fit_covariance_surface(
        data,
        residuals,
        grid,
        cov_n_basis,
        smoothing_parameter,
    )

    if isinstance(noise_variance, str):
        # It can only be "auto" at this point.
        noise_variance_value = _estimate_noise_variance(
            data,
            residuals,
            covariance_grid,
            grid,
            cov_n_basis,
            smoothing_parameter,
        )
    else:
        noise_variance_value = float(noise_variance)

    eigenvalues, eigenvectors = _project_to_psd(covariance_grid)
    if len(eigenvalues) == 0:
        raise ValueError(
            "The estimated covariance surface has no positive "
            "eigenvalue: the data do not contain enough signal to "
            "estimate a covariance kernel.",
        )

    covariance_kernel = EigenExpansion(
        grid_points=grid,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        domain_range=(data.domain_range,),
    )

    location = FDataGrid(
        data_matrix=mean_grid[np.newaxis, :, np.newaxis],
        grid_points=grid,
        domain_range=X.domain_range,
    )

    return location, covariance_kernel, noise_variance_value
