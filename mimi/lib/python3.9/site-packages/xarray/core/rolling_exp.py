from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic

import numpy as np
from packaging.version import Version

from xarray.core.computation import apply_ufunc
from xarray.core.options import _get_keep_attrs
from xarray.core.pdcompat import count_not_none
from xarray.core.types import T_DataWithCoords

try:
    import numbagg
    from numbagg import move_exp_nanmean, move_exp_nansum

    _NUMBAGG_VERSION: Version | None = Version(numbagg.__version__)
except ImportError:
    _NUMBAGG_VERSION = None


def _get_alpha(
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> float:
    # pandas defines in terms of com (converting to alpha in the algo)
    # so use its function to get a com and then convert to alpha

    com = _get_center_of_mass(com, span, halflife, alpha)
    return 1 / (1 + com)


def _get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    """
    Vendored from pandas.core.window.common._get_center_of_mass

    See licenses/PANDAS_LICENSE for the function's license
    """
    valid_count = count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2.0
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1.0 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


class RollingExp(Generic[T_DataWithCoords]):
    """
    Exponentially-weighted moving window object.
    Similar to EWM in pandas

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : mapping of hashable to int (or float for alpha type)
        A mapping from the name of the dimension to create the rolling
        exponential window along (e.g. `time`) to the size of the moving window.
    window_type : {"span", "com", "halflife", "alpha"}, default: "span"
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(
        self,
        obj: T_DataWithCoords,
        windows: Mapping[Any, int | float],
        window_type: str = "span",
        min_weight: float = 0.0,
    ):
        if _NUMBAGG_VERSION is None:
            raise ImportError(
                "numbagg >= 0.2.1 is required for rolling_exp but currently numbagg is not installed"
            )
        elif _NUMBAGG_VERSION < Version("0.2.1"):
            raise ImportError(
                f"numbagg >= 0.2.1 is required for rolling_exp but currently version {_NUMBAGG_VERSION} is installed"
            )
        elif _NUMBAGG_VERSION < Version("0.3.1") and min_weight > 0:
            raise ImportError(
                f"numbagg >= 0.3.1 is required for `min_weight > 0` within `.rolling_exp` but currently version {_NUMBAGG_VERSION} is installed"
            )

        self.obj: T_DataWithCoords = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})
        self.min_weight = min_weight
        # Don't pass min_weight=0 so we can support older versions of numbagg
        kwargs = dict(alpha=self.alpha, axis=-1)
        if min_weight > 0:
            kwargs["min_weight"] = min_weight
        self.kwargs = kwargs

    def mean(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving average.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)>
        array([1.        , 1.        , 1.69230769, 1.9       , 1.96694215])
        Dimensions without coordinates: x
        """

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            move_exp_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def sum(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving sum.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").sum()
        <xarray.DataArray (x: 5)>
        array([1.        , 1.33333333, 2.44444444, 2.81481481, 2.9382716 ])
        Dimensions without coordinates: x
        """

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            move_exp_nansum,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def std(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving standard deviation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").std()
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 0.67936622, 0.42966892, 0.25389527])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().std(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanstd,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def var(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving variance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").var()
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 0.46153846, 0.18461538, 0.06446281])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().var(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanvar,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def cov(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving covariance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").cov(da**2)
        <xarray.DataArray (x: 5)>
        array([       nan, 0.        , 1.38461538, 0.55384615, 0.19338843])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().cov(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nancov,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def corr(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving correlation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").corr(da.shift(x=1))
        <xarray.DataArray (x: 5)>
        array([       nan,        nan,        nan, 0.4330127 , 0.48038446])
        Dimensions without coordinates: x
        """

        if _NUMBAGG_VERSION is None or _NUMBAGG_VERSION < Version("0.4.0"):
            raise ImportError(
                f"numbagg >= 0.4.0 is required for rolling_exp().cov(), currently {_NUMBAGG_VERSION} is installed"
            )
        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nancorr,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)
