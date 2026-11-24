"""Functional Principal Component Analysis Module."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np
import scipy.integrate
import scipy.linalg
from scipy.linalg import solve_triangular
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis, FDataBasis, _GridBasis
from ...representation.grid import FDataGrid
from ...representation.irregular import FDataIrregular
from ...typing._numpy import ArrayLike, NDArrayFloat

if TYPE_CHECKING:
    from ...exploratory.stats.covariance import CovarianceEstimator

Function = TypeVar("Function", bound=FData)
WeightsCallable = Callable[[np.ndarray], np.ndarray]


class FPCA(  # noqa: WPS230 (too many public attributes)
    InductiveTransformerMixin[FData, NDArrayFloat, object],
    BaseEstimator,
):
    r"""
    Principal component analysis.

    Class that implements functional principal component analysis for both
    basis and grid representations of the data. The parameters are shared
    when fitting a FDataBasis or FDataGrid, except for ``components_basis``.

    For more information about the implementation of the computation of the
    first principal components see Ramsay and Silverman (chapter 8, section 
    4)\ :footcite:`ramsay+silverman_2005`.

    Parameters:
        n_components: Number of principal components to keep from
            functional principal component analysis.

            .. versionchanged:: 0.9
               In future versions, it will default to the maximum number
               of components that can be extracted.
               Currently, it still defaults to 3 but do not assume this
               behavior as it will change.
        centering: Set to ``False`` when the functional data is already known
            to be centered and there is no need to center it. Otherwise,
            the mean of the functional data object is calculated and the data
            centered before fitting . Defaults to ``True``.
        regularization: Regularization object to be applied.
        components_basis: The basis in which we want the principal
            components. We can use a different basis than the basis contained
            in the passed FDataBasis object. This parameter is only used when
            fitting a FDataBasis.
        covariance_estimator: Estimator of the covariance operator of the
            process, used to analyze sparse and noisy observations. It must
            be ``None`` (the default) or an estimator following the
            protocol of
            :class:`~skfda.exploratory.stats.covariance.EmpiricalCovariance`
            with ``sparse=True``: once fitted with the training
            :class:`~skfda.representation.FDataIrregular` curves, it must
            expose a one-sample ``location_`` FDataGrid with the mean
            function, a callable ``covariance_`` kernel of the latent
            process (without measurement noise) and a nonnegative
            ``noise_variance_``. The estimator is cloned and fitted only
            with the training curves inside :meth:`fit`. When this
            parameter is used, the data must be a
            :class:`~skfda.representation.FDataIrregular` object and only
            ``centering=True``, ``regularization=None``,
            ``components_basis=None`` and ``_weights=None`` are supported.
            The scores computed by :meth:`transform` are then the mean of
            the principal component scores of each new curve conditioned
            on its noisy observations, under the fitted Gaussian process
            model, and :meth:`inverse_transform` reconstructs latent
            curves on the estimator's grid.
        rank_rtol: Relative tolerance used, only when
            ``covariance_estimator`` is provided, for the floating point
            rank decisions: eigenvalues of the discretized covariance
            operator are considered positive when they are larger than
            ``rank_rtol`` times the largest one, and, for noiseless
            models, observations are considered compatible with the model
            when the relative residual of the (possibly rank-deficient)
            observation system is at most ``rank_rtol``.

    Attributes:
        components\_: this contains the principal components.
        explained_variance\_ : The amount of variance explained by
            each of the selected components.
        explained_variance_ratio\_ : this contains the percentage
            of variance explained by each principal component. When
            ``covariance_estimator`` is used, the denominator is the sum
            of all the nonnegative eigenvalues of the discretized
            covariance operator, and thus does not depend on the number
            of components kept.
        singular_values\_: The singular values corresponding to each of the
            selected components.
        mean\_: mean of the train data.
        covariance_estimator\_: the covariance estimator fitted with the
            training curves, when ``covariance_estimator`` is provided.

    Examples:
        Construct an artificial FDataBasis object and run FPCA with this
        object. The resulting principal components are not compared because
        there are several equivalent possibilities.

        >>> import skfda
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> basis = skfda.representation.basis.MonomialBasis(
        ...     domain_range=(0,1), n_basis=2
        ... )
        >>> basis_fd = fd.to_basis(basis)
        >>> fpca_basis = FPCA(2)
        >>> fpca_basis = fpca_basis.fit(basis_fd)

        In this example we apply discretized functional PCA with some simple
        data to illustrate the usage of this class. We initialize the
        FPCA object, fit the artificial data and obtain the scores.
        The results are not tested because there are several equivalent
        possibilities.

        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fpca_grid = FPCA(2)
        >>> fpca_grid = fpca_grid.fit(fd)

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        centering: bool = True,
        regularization: L2Regularization[FData] | None = None,
        components_basis: Basis | None = None,
        _weights: ArrayLike | WeightsCallable | None = None,
        covariance_estimator: CovarianceEstimator[FDataIrregular] | None = None,
        rank_rtol: float = 1e-10,
    ) -> None:

        if n_components is None:
            warnings.warn(
                "The default value of n_components will change in a future "
                "version to the maximum number of components that can be "
                "extracted. Update your code to specify explicitly the "
                "number of components to avoid this warning.",
                DeprecationWarning,
            )
            n_components = 3
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self._weights = _weights
        self.components_basis = components_basis
        self.covariance_estimator = covariance_estimator
        self.rank_rtol = rank_rtol

    def _center_if_necessary(
        self,
        X: Function,
        *,
        learn_mean: bool = True,
    ) -> Function:

        if learn_mean:
            self.mean_ = X.mean()

        return X - self.mean_ if self.centering else X

    def _fit_basis(
        self,
        X: FDataBasis,
        y: object = None,
    ) -> FPCA:
        """
        Compute the first n_components principal components and saves them.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        """
        # the maximum number of components is established by the target basis
        # if the target basis is available.
        n_basis = (
            self.components_basis.n_basis
            if self.components_basis
            else X.basis.n_basis
        )
        # necessary in inverse_transform
        self.n_samples_ = X.n_samples

        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError(
                "The sample size must be bigger than the "
                "number of components",
            )

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the basis
        if self.n_components > n_basis:
            raise AttributeError(
                "The number of components should be "
                "smaller than the number of attributes of "
                "target principal components' basis.",
            )

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # setup principal component basis if not given
        components_basis = self.components_basis
        if components_basis is not None:
            # First fix domain range if not already done
            components_basis = components_basis.copy(
                domain_range=X.basis.domain_range,
            )
            g_matrix = components_basis.gram_matrix()
            # The matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            j_matrix = X.basis.inner_product_matrix(components_basis)
        else:
            # If no other basis is specified we use the same basis as the
            # passed FDataBasis object
            components_basis = X.basis.copy()
            g_matrix = components_basis.gram_matrix()
            j_matrix = g_matrix

        self._X_basis = X.basis
        self._j_matrix = j_matrix

        # Apply regularization / penalty if applicable
        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(components_basis,),
            regularization_parameter=1,
            regularization=self.regularization,
        )

        # apply regularization
        if regularization_matrix is not None:
            # using += would have a different behavior
            g_matrix = g_matrix + regularization_matrix  # noqa: WPS350

        # obtain triangulation using cholesky
        l_matrix = np.linalg.cholesky(g_matrix)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = solve_triangular(
            l_matrix,
            np.transpose(j_matrix),
            lower=True,
        )

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
        final_matrix = X.coefficients @ np.transpose(l_inv_j_t)

        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = solve_triangular(
            np.transpose(l_matrix),
            np.transpose(pca.components_),
            lower=False,
        )

        self.explained_variance_ratio_: NDArrayFloat = (
            pca.explained_variance_ratio_
        )
        self.explained_variance_: NDArrayFloat = pca.explained_variance_
        self.singular_values_: NDArrayFloat = pca.singular_values_
        self.components_: FData = X.copy(
            basis=components_basis,
            coefficients=component_coefficients.T,
            sample_names=(None,) * self.n_components,
        )

        return self

    def _transform_basis(
        self,
        X: FDataBasis,
        y: object = None,
    ) -> NDArrayFloat:
        """Compute the n_components first principal components score.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        if X.basis != self._X_basis:
            raise ValueError(
                "The basis used in fit is different from "
                "the basis used in transform.",
            )

        # in this case it is the inner product of our data with the components
        return (  # type: ignore[no-any-return]
            X.coefficients @ self._j_matrix
            @ self.components_.coefficients.T
        )

    def _fit_grid(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> FPCA:
        r"""
        Compute the n_components first principal components and saves them.

        The eigenvalues associated with these principal
        components are also saved. For more details about how it is implemented
        please view the referenced book, chapter 8.

        In summary, we are performing standard multivariate PCA over
        :math:`\mathbf{X} \mathbf{W}^{1/2}` where :math:`\mathbf{X}` is the
        data matrix and :math:`\mathbf{W}` is the weight matrix (this matrix
        defines the numerical integration). By default the weight matrix is
        obtained using the trapezoidal rule.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self.

        """
        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError(
                "The sample size must be bigger than the "
                "number of components",
            )

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the funcional data object
        if self.n_components > X.data_matrix.shape[1]:
            raise AttributeError(
                "The number of components should be "
                "smaller than the number of discretization "
                "points of the functional data object.",
            )

        # data matrix initialization
        fd_data = X.data_matrix.reshape(X.data_matrix.shape[:-1])

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # necessary for inverse_transform
        self.n_samples_ = n_samples

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # establish weights for each point of discretization
        if self._weights is None:
            # grid_points is a list with one array in the 1D case
            identity = np.eye(len(X.grid_points[0]))
            self._weights = scipy.integrate.simpson(
                y=identity,
                x=X.grid_points[0],
            )
        elif callable(self._weights):
            self._weights = self._weights(X.grid_points[0])
            # if its a FDataGrid then we need to reduce the dimension to 1-D
            # array
            if isinstance(self._weights, FDataGrid):
                self._weights = np.squeeze(self._weights.data_matrix)
        else:
            self._weights = self._weights

        weights_matrix = np.diag(self._weights)

        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(_GridBasis(grid_points=X.grid_points),),
            regularization_parameter=1,
            regularization=self.regularization,
        )

        # See issue #497 for more information about this approach
        factorization_matrix = weights_matrix.astype(float)
        if self.regularization is not None:
            factorization_matrix += regularization_matrix

        # Tranpose of the Cholesky decomposition
        Lt = np.linalg.cholesky(factorization_matrix).T

        new_data_matrix = fd_data @ weights_matrix
        new_data_matrix = np.linalg.solve(Lt.T, new_data_matrix.T).T

        pca = PCA(n_components=self.n_components)
        pca.fit(new_data_matrix)

        components = pca.components_
        components = np.linalg.solve(Lt, pca.components_.T).T

        self.components_ = X.copy(
            data_matrix=components,
            sample_names=(None,) * self.n_components,
        )

        self.explained_variance_ratio_ = (
            pca.explained_variance_ratio_
        )
        self.explained_variance_ = pca.explained_variance_
        self.singular_values_ = pca.singular_values_

        return self

    def _transform_grid(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components score.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors

        return (  # type: ignore[no-any-return]
            X.data_matrix.reshape(X.data_matrix.shape[:-1])
            * self._weights
            @ np.transpose(
                self.components_.data_matrix.reshape(
                    self.components_.data_matrix.shape[:-1],
                ),
            )
        )

    def _check_sparse_parameters(self) -> None:
        """Check that the parameters are compatible with the sparse path."""
        unsupported = []
        if not self.centering:
            unsupported.append("centering=False")
        if self.regularization is not None:
            unsupported.append("regularization is not None")
        if self.components_basis is not None:
            unsupported.append("components_basis is not None")
        if self._weights is not None:
            unsupported.append("_weights is not None")
        if unsupported:
            raise NotImplementedError(
                "FPCA with a covariance estimator only supports "
                "centering=True, regularization=None, "
                "components_basis=None and _weights=None, but "
                f"{', '.join(unsupported)} was provided.",
            )

    def _fit_irregular(
        self,
        X: FDataIrregular,
        y: object = None,
    ) -> FPCA:
        r"""Fit FPCA on sparse noisy curves through a covariance estimator.

        The mean, covariance kernel and noise variance are obtained from
        the (cloned) covariance estimator, fitted only with the training
        curves. The principal components are the eigenfunctions of the
        discretized covariance operator

        .. math::
            (T f)(s) = \int K(s, t) f(t) dt

        approximated on the estimator's grid with the quadrature weights
        used also for the integration inner product of the components.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        """
        self._check_sparse_parameters()

        if not (0.0 < self.rank_rtol < 1.0):
            raise ValueError(
                f"rank_rtol must be in the (0, 1) interval, but it is "
                f"{self.rank_rtol}.",
            )

        if not isinstance(X, FDataIrregular):
            raise ValueError(
                "FPCA with a covariance estimator requires a "
                f"FDataIrregular object, but {type(X).__name__} was "
                "provided.",
            )
        if X.dim_domain != 1 or X.dim_codomain != 1:
            raise NotImplementedError(
                "FPCA with a covariance estimator is only implemented "
                "for functions with one-dimensional domain and "
                f"codomain, but dim_domain={X.dim_domain} and "
                f"dim_codomain={X.dim_codomain}.",
            )

        estimator = clone(self.covariance_estimator)
        estimator.fit(X)
        self.covariance_estimator_ = estimator

        for attribute in ("location_", "covariance_", "noise_variance_"):
            if not hasattr(estimator, attribute):
                raise TypeError(
                    "The covariance estimator does not follow the "
                    "required protocol: after fitting it must expose "
                    "location_, covariance_ and noise_variance_, but "
                    f"{attribute} is missing.",
                )

        mean = estimator.location_
        if (
            not isinstance(mean, FDataGrid)
            or mean.n_samples != 1
            or mean.dim_domain != 1
            or mean.dim_codomain != 1
        ):
            raise TypeError(
                "The covariance estimator must provide the mean as a "
                "one-sample FDataGrid with one-dimensional domain and "
                "codomain.",
            )

        noise_variance = float(estimator.noise_variance_)
        if not np.isfinite(noise_variance) or noise_variance < 0:
            raise ValueError(
                "The covariance estimator must provide a finite "
                f"nonnegative noise_variance_, but it is "
                f"{estimator.noise_variance_}.",
            )

        self.mean_ = mean
        self.n_samples_ = X.n_samples

        grid_points = np.asarray(mean.grid_points[0], dtype=np.float64)
        n_grid = len(grid_points)

        if self.n_components > n_grid:
            raise ValueError(
                "The number of components should not be larger than "
                "the number of points of the estimation grid "
                f"({n_grid}).",
            )

        cov_matrix = np.asarray(
            estimator.covariance_(grid_points, grid_points),
            dtype=np.float64,
        )
        expected_shape = (n_grid, n_grid)
        if cov_matrix.shape != expected_shape:
            raise ValueError(
                "The covariance kernel evaluated on the estimation "
                f"grid must have shape {expected_shape}, but it has "
                f"shape {cov_matrix.shape}.",
            )
        if not np.all(np.isfinite(cov_matrix)):
            raise ValueError(
                "The covariance kernel returned non-finite values on "
                "the estimation grid.",
            )
        # Remove the asymmetry introduced by floating point roundoff.
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # Quadrature weights of the integration inner product. Simpson
        # weights are used when they are positive (in particular for
        # equally spaced grids); otherwise the trapezoidal rule, whose
        # weights are always positive, is used instead.
        weights = scipy.integrate.simpson(
            y=np.eye(n_grid),
            x=grid_points,
        )
        if np.any(weights <= 0):
            weights = scipy.integrate.trapezoid(
                y=np.eye(n_grid),
                x=grid_points,
            )
        self._quadrature_weights = weights

        # Eigen decomposition of the discretized covariance operator:
        # W^{1/2} K W^{1/2}, whose eigenvectors scaled by W^{-1/2} are
        # the components, orthonormal under the integration inner
        # product.
        sqrt_weights = np.sqrt(weights)
        operator_matrix = (
            sqrt_weights[:, np.newaxis]
            * cov_matrix
            * sqrt_weights[np.newaxis, :]
        )
        eigenvalues, eigenvectors = scipy.linalg.eigh(operator_matrix)

        smallest = eigenvalues[0]
        largest = eigenvalues[-1]
        if smallest < -self.rank_rtol * max(abs(largest), 1.0):
            raise ValueError(
                "The covariance kernel provided by the estimator is "
                "not positive semi-definite on the estimation grid.",
            )
        rank_tolerance = self.rank_rtol * max(largest, 0.0)
        positive_rank = int(np.sum(eigenvalues > rank_tolerance))
        if self.n_components > positive_rank:
            raise ValueError(
                "The number of components should not be larger than "
                "the numerical positive rank of the covariance "
                f"operator ({positive_rank}).",
            )

        # Eigenvalues are returned in ascending order.
        keep = slice(n_grid - self.n_components, n_grid)
        self.explained_variance_ = eigenvalues[keep][::-1]
        total_variance = np.sum(eigenvalues[eigenvalues > 0])
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )
        self.singular_values_ = np.sqrt(self.explained_variance_)

        components = (
            eigenvectors[:, keep][:, ::-1] / sqrt_weights[:, np.newaxis]
        )
        self.components_ = FDataGrid(
            data_matrix=components.T[:, :, np.newaxis],
            grid_points=grid_points,
            domain_range=mean.domain_range,
            sample_names=(None,) * self.n_components,
        )

        return self

    def _transform_irregular(
        self,
        X: FDataIrregular,
        y: object = None,
    ) -> NDArrayFloat:
        r"""Compute conditional principal component scores of sparse curves.

        The scores are the mean of the principal component scores of
        each curve conditioned on its noisy observations, under the
        fitted Gaussian process model. If :math:`\xi_k` is the k-th
        score, :math:`\mu` the mean, :math:`K` the covariance kernel of
        the latent process and :math:`\sigma^2` the noise variance, for
        observations :math:`y` of a curve at times :math:`t_1, \dots,
        t_{n_i}`,

        .. math::
            E[\xi_k \mid y] = \lambda_k \, \phi_k(t)^\top
            \left( K(t, t) + \sigma^2 I \right)^{-1} (y - \mu(t))

        where :math:`\lambda_k` and :math:`\phi_k` are the eigenvalues
        and eigenfunctions of the covariance operator. When
        :math:`\sigma^2 = 0` the observation covariance can be singular:
        the system is then solved in the least squares sense with the
        rank decided by ``rank_rtol``, and observations incompatible
        with the model support are rejected.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        check_is_fitted(self, "covariance_estimator_")

        if not isinstance(X, FDataIrregular):
            raise ValueError(
                "FPCA fitted with a covariance estimator can only "
                f"transform FDataIrregular objects, but "
                f"{type(X).__name__} was provided.",
            )
        if X.dim_domain != 1 or X.dim_codomain != 1:
            raise NotImplementedError(
                "FPCA fitted with a covariance estimator only supports "
                "functions with one-dimensional domain and codomain.",
            )

        estimator = self.covariance_estimator_
        covariance = estimator.covariance_
        noise_variance = float(estimator.noise_variance_)

        domain_range, = self.mean_.domain_range
        domain_start, domain_end = domain_range

        eigenvalues = self.explained_variance_

        points = np.asarray(X.points[:, 0], dtype=np.float64)
        values = np.asarray(X.values[:, 0], dtype=np.float64)

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
            raise ValueError(
                "The observation points and values must be finite.",
            )
        if np.any(points < domain_start) or np.any(points > domain_end):
            raise ValueError(
                "The observation points must lie inside the domain of "
                f"the fitted model [{domain_start}, {domain_end}].",
            )

        end_indices = np.concatenate(
            (X.start_indices[1:], [len(points)]),
        )

        scores = np.empty((X.n_samples, self.n_components))
        for sample, (start, end) in enumerate(
            zip(X.start_indices, end_indices),
        ):
            curve_points = points[start:end]
            curve_values = values[start:end]

            if len(curve_points) == 0:
                # Without observations, the conditional scores are the
                # prior ones, whose mean is zero.
                scores[sample] = 0.0
                continue

            mean_obs = self.mean_(curve_points)[0, :, 0]
            components_obs = self.components_(curve_points)[..., 0].T
            centered = curve_values - mean_obs

            obs_covariance = np.asarray(
                covariance(curve_points, curve_points),
                dtype=np.float64,
            )
            expected_shape = (len(curve_points), len(curve_points))
            if obs_covariance.shape != expected_shape:
                raise ValueError(
                    "The covariance kernel of the estimator returned "
                    f"an array of shape {obs_covariance.shape} for a "
                    f"curve with {len(curve_points)} observations, but "
                    f"shape {expected_shape} was expected.",
                )
            if not np.all(np.isfinite(obs_covariance)):
                raise ValueError(
                    "The covariance kernel of the estimator returned "
                    "non-finite values at the observation points.",
                )
            obs_covariance = (obs_covariance + obs_covariance.T) / 2
            obs_covariance = obs_covariance + noise_variance * np.eye(
                len(curve_points),
            )

            coefficients = self._solve_observation_system(
                obs_covariance,
                centered,
                noise_variance=noise_variance,
            )

            scores[sample] = eigenvalues * (components_obs.T @ coefficients)

        return scores

    def _solve_observation_system(
        self,
        obs_covariance: NDArrayFloat,
        centered: NDArrayFloat,
        *,
        noise_variance: float,
    ) -> NDArrayFloat:
        """Solve the observation covariance system of one curve.

        For noisy models the system is positive definite and is solved
        with the Cholesky decomposition. For noiseless models it can be
        singular and is solved with a truncated pseudo-inverse, checking
        that the observations are compatible with the model support.
        """
        if noise_variance > 0:
            try:
                cho_factor = scipy.linalg.cho_factor(obs_covariance)
                return scipy.linalg.cho_solve(cho_factor, centered)
            except scipy.linalg.LinAlgError:
                # Fall back to the pseudo-inverse path below.
                pass

        solution, residual_norm = self._lstsq_observation_system(
            obs_covariance,
            centered,
        )

        residual_reference = max(np.linalg.norm(centered), 1e-300)
        if residual_norm > self.rank_rtol * residual_reference:
            raise ValueError(
                "The observations are not compatible with the fitted "
                "Gaussian process model: they lie outside the support "
                "of the noiseless observation covariance (relative "
                f"residual {residual_norm / residual_reference:.3e} "
                f"> rank_rtol={self.rank_rtol:.1e}).",
            )

        return solution

    def _lstsq_observation_system(
        self,
        obs_covariance: NDArrayFloat,
        centered: NDArrayFloat,
    ) -> tuple[NDArrayFloat, float]:
        """Solve a possibly singular observation system.

        The rank is decided with the publicly documented ``rank_rtol``
        relative tolerance on the singular values.

        Returns:
            The minimum-norm solution and the residual norm.

        """
        solution, _, rank, singular_values = scipy.linalg.lstsq(
            obs_covariance,
            centered,
            cond=self.rank_rtol,
            lapack_driver="gelsd",
        )
        if rank == 0:
            residual_norm = np.linalg.norm(centered)
        else:
            residual_norm = np.linalg.norm(
                obs_covariance @ solution - centered,
            )
        return solution, residual_norm

    def fit(
        self,
        X: FData,
        y: object = None,
    ) -> FPCA:
        """
        Compute the n_components first principal components and saves them.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        """
        if self.covariance_estimator is not None:
            return self._fit_irregular(X, y)  # type: ignore[arg-type]
        if isinstance(X, FDataGrid):
            return self._fit_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._fit_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components scores.

        Args:
            X: The functional data object to be analysed.
            y: Only present because of fit function convention

        Returns:
            Principal component scores.

        """
        if self.covariance_estimator is not None:
            return self._transform_irregular(X, y)  # type: ignore[arg-type]

        X = self._center_if_necessary(X, learn_mean=False)

        if isinstance(X, FDataGrid):
            return self._transform_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._transform_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.

        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self,
        pc_scores: NDArrayFloat,
    ) -> FData:
        """
        Compute the recovery from the fitted principal components scores.

        In other words,
        it maps ``pc_scores``, from the fitted functional PCs' space,
        back to the input functional space.
        ``pc_scores`` might be an array returned by ``transform`` method.

        Args:
            pc_scores: ndarray (n_samples, n_components).

        Returns:
            A FData object.

        """
        # check the instance is fitted.

        # input format check:
        if isinstance(pc_scores, np.ndarray):
            if pc_scores.ndim == 1:
                pc_scores = pc_scores[np.newaxis, :]

            if pc_scores.shape[1] != self.n_components:
                raise AttributeError(
                    "pc_scores must be a numpy array "
                    "with n_samples rows and n_components columns.",
                )
        else:
            raise AttributeError("pc_scores is not a numpy array.")

        # inverse_transform is slightly different whether
        # .fit was applied to FDataGrid or FDataBasis object
        # Does not work (boundary problem in x_hat and bias reconstruction)
        if isinstance(self.components_, FDataGrid):

            additional_args = {
                "data_matrix": np.einsum(
                    "nc,c...->n...",
                    pc_scores,
                    self.components_.data_matrix,
                ),
            }

        elif isinstance(self.components_, FDataBasis):

            additional_args = {
                "coefficients": pc_scores @ self.components_.coefficients,
            }

        return (
            self.mean_.copy(
                **additional_args,
                sample_names=(None,) * len(pc_scores),
            )
            + self.mean_
        )
