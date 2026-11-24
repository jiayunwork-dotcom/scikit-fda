from __future__ import annotations

from typing import Callable, Literal, TypeVar

from ....representation import FData
from ....typing._numpy import ArrayLike, NDArrayFloat
from ._base import CovarianceEstimator
from ._sparse import fit_sparse_covariance

Input = TypeVar("Input", bound=FData)


class EmpiricalCovariance(
    CovarianceEstimator[Input],
):
    r"""Empirical covariance estimator.

    By default (``sparse=False``) the location is the sample mean and the
    covariance is the sample covariance function of the observed curves,
    which requires densely observed (or already smoothed) data.

    With ``sparse=True`` the estimator instead works directly on sparse,
    irregularly sampled and noisily measured curves, represented as a
    :class:`~skfda.representation.FDataIrregular` object, and separates
    the latent process variation from the independent measurement noise.

    Parameters:
        assume_centered: If ``True``, the location is assumed to be zero
            and the data is not centered before computing the covariance.
        sparse: If ``True``, use the sparse estimator for irregularly
            sampled noisy curves. If ``False`` (the default), use the
            classical sample estimators.
        grid_points: Evaluation grid on which the mean and the covariance
            operator are estimated when ``sparse=True``. It must contain
            at least 3 finite, strictly increasing points whose extremes
            coincide with the declared domain range of the data. Ignored
            when ``sparse=False``.
        noise_variance: Variance of the independent measurement noise.
            If ``"auto"`` (the default), it is estimated from the data.
            It can also be a finite nonnegative real number (in
            particular, ``0`` models noiseless observations). Only used
            when ``sparse=True``.
        mean_n_basis: Number of cubic B-spline basis functions of the
            penalized smoother of the mean function. Larger values follow
            the data more closely (less smoothing); smaller values give
            smoother estimates. Only used when ``sparse=True``.
        cov_n_basis: Number of cubic B-spline basis functions per
            dimension of the penalized tensor-product smoother of the
            covariance surface. Larger values allow more flexible
            covariance surfaces; smaller values give smoother estimates.
            Only used when ``sparse=True``.
        smoothing_parameter: Nonnegative relative penalty of the
            smoothers used when ``sparse=True``. The penalty of each
            smoother is scaled by the ratio of the traces of its Gram and
            penalty matrices, so the parameter is roughly comparable
            across sample sizes and basis sizes. ``0`` gives unpenalized
            regression splines, whose smoothness is then controlled only
            by the number of basis functions.

    Attributes:
        location\_: Estimated mean function. When ``sparse=True`` it is a
            one-sample :class:`~skfda.representation.FDataGrid` on
            ``grid_points``.
        covariance\_: Estimated covariance of the latent process, as a
            callable kernel. When ``sparse=True`` it does not contain the
            independent measurement noise, and it can be evaluated at any
            points of the domain, not only at ``grid_points``.
        noise_variance\_: Estimated (or fixed, if provided) variance of
            the independent measurement noise. Only set when
            ``sparse=True``.
        grid_points\_: Evaluation grid on which the mean and the
            covariance operator are estimated. Only set when
            ``sparse=True``.

    Notes:
        When ``sparse=True``, the input must be a
        :class:`~skfda.representation.FDataIrregular` object with
        one-dimensional domain and codomain, containing at least 3 curves
        with at least 2 finite observations each. The observation times
        of each curve must be strictly increasing and lie inside the
        declared domain range, but different curves do not need to share
        any observation time.

        The estimation follows the PACE methodology (Yao, Müller and
        Wang, 2005, *Functional Data Analysis for Sparse Longitudinal
        Data*, JASA 100: 577-590): the mean is smoothed from the pooled
        observations, and the covariance surface is smoothed from the
        pooled within-curve cross-products of the residuals, excluding
        the diagonal (which is contaminated by the measurement noise).
        Only pairs of observations of the *same* curve are multiplied,
        so the estimator exploits the within-curve covariation without
        requiring common observation times across curves. The noise
        variance is the difference between the smoothed diagonal
        of the raw covariance (which contains the noise) and the
        diagonal of the smoothed surface (which does not), averaged
        over the interior of the domain to avoid the less reliable
        boundary regions.

        Two ways of organizing the smoothing were considered:

        * Local polynomial smoothers with data-driven bandwidths, as in
          the original PACE proposal. They are locally adaptive, but
          require a costly and fragile bandwidth selection, have boundary
          bias, evaluate each grid point with an independent weighted
          regression (quadratic cost in the number of pooled pairs) and
          still need a post-hoc positive semi-definite projection.
        * Global penalized B-spline smoothers (implemented here). The
          estimation reduces to a few linear systems whose size depends
          only on the number of basis functions, not on the number of
          observations, so the cost is linear in the number of pooled
          pairs and the memory footprint is bounded. The price is a
          global (not locally adaptive) smoothness assumption: with
          ``cov_n_basis`` too small the covariance surface is
          over-smoothed (bias), and with ``cov_n_basis`` too large the
          estimate is noisy and the normal equations can become
          ill-conditioned in regions with little observational support,
          in which case the fit fails with an explicit error instead of
          returning a degenerate kernel.

        In both cases the smoothed covariance surface is not guaranteed
        to be positive semi-definite. Here it is projected onto the
        positive semi-definite cone by truncating the negative
        eigenvalues of its discretization on ``grid_points``, and the
        delivered kernel is the corresponding eigen-expansion with
        linearly interpolated eigenfunctions. The kernel is therefore
        positive semi-definite on *every* finite set of domain points,
        including points outside the training grid; a kernel that only
        fixes the eigenvalues on the training grid would not be valid
        off-grid and is not delivered.

    Examples:
        >>> import numpy as np
        >>> from skfda.exploratory.stats.covariance import (
        ...     EmpiricalCovariance,
        ... )
        >>> from skfda.representation import FDataGrid
        >>> grid_points = np.linspace(0, 1, 5)
        >>> data_matrix = np.array([
        ...     [0.0, 1.0, 2.0, 1.0, 0.0],
        ...     [0.0, 2.0, 4.0, 2.0, 0.0],
        ...     [0.0, 0.5, 1.0, 0.5, 0.0],
        ... ])
        >>> fd = FDataGrid(data_matrix, grid_points=grid_points)
        >>> estimator = EmpiricalCovariance()
        >>> estimator.fit(fd)
        EmpiricalCovariance()
        >>> estimator.covariance_([0.0, 0.5], [0.0, 0.5])
        array(...)

    """

    covariance_: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]

    def __init__(
        self,
        *,
        assume_centered: bool = False,
        sparse: bool = False,
        grid_points: ArrayLike | None = None,
        noise_variance: Literal["auto"] | float = "auto",
        mean_n_basis: int = 12,
        cov_n_basis: int = 10,
        smoothing_parameter: float = 1e-4,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.sparse = sparse
        self.grid_points = grid_points
        self.noise_variance = noise_variance
        self.mean_n_basis = mean_n_basis
        self.cov_n_basis = cov_n_basis
        self.smoothing_parameter = smoothing_parameter

    def _fit_sparse(self, X: Input, y: object = None) -> EmpiricalCovariance:
        """Fit the sparse estimator on irregularly sampled curves."""
        (
            self.location_,
            self.covariance_,
            self.noise_variance_,
        ) = fit_sparse_covariance(
            X,  # type: ignore[arg-type]
            grid_points=self.grid_points,
            noise_variance=self.noise_variance,
            assume_centered=self.assume_centered,
            mean_n_basis=self.mean_n_basis,
            cov_n_basis=self.cov_n_basis,
            smoothing_parameter=self.smoothing_parameter,
        )
        self.grid_points_ = self.location_.grid_points[0]
        return self

    def fit(self, X: Input, y: object = None) -> EmpiricalCovariance[Input]:
        """Fit the covariance estimator.

        Args:
            X: Functional data whose covariance is estimated. A
                :class:`~skfda.representation.FDataIrregular` object is
                required when ``sparse=True``.
            y: Ignored.

        Returns:
            self

        """
        if self.sparse:
            return self._fit_sparse(X, y)

        super().fit(X, y)
        self.covariance_ = X.cov()
        return self
