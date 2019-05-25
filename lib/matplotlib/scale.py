import inspect
import textwrap

import numpy as np
from numpy import ma

from matplotlib import cbook, docstring, rcParams
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter,
    NullLocator, LogLocator, AutoLocator, AutoMinorLocator,
    SymmetricalLogLocator, LogitLocator)
from matplotlib.transforms import Transform, IdentityTransform


class ScaleBase(object):
    """
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Any subclasses will want to override:

      - :attr:`name`
      - :meth:`get_transform`
      - :meth:`set_default_locators_and_formatters`

    And optionally:
      - :meth:`limit_range_for_scale`
    """

    def __init__(self, axis, **kwargs):
        r"""
        Construct a new scale.

        Notes
        -----
        The following note is for scale implementors.

        For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
        object as first argument.  However, this argument should not
        be used: a single scale object should be usable by multiple
        `~matplotlib.axis.Axis`\es at the same time.
        """

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` object
        associated with this scale.
        """
        raise NotImplementedError()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the :class:`~matplotlib.ticker.Locator` and
        :class:`~matplotlib.ticker.Formatter` objects on the given
        axis to match this scale.
        """
        raise NotImplementedError()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Returns the range *vmin*, *vmax*, possibly limited to the
        domain supported by this scale.

        *minpos* should be the minimum positive value in the data.
         This is used by log scales to determine a minimum value.
        """
        return vmin, vmax


class LinearScale(ScaleBase):
    """
    The default linear scale.
    """

    name = 'linear'

    def __init__(self, axis, **kwargs):
        # This method is present only to prevent inheritance of the base class'
        # constructor docstring, which would otherwise end up interpolated into
        # the docstring of Axis.set_scale.
        """
        """
        super().__init__(axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults for
        linear scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and rcParams['xtick.minor.visible']
            or axis.axis_name == 'y' and rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())

    def get_transform(self):
        """
        The transform for linear scaling is just the
        :class:`~matplotlib.transforms.IdentityTransform`.
        """
        return IdentityTransform()


class FuncTransform(Transform):
    """
    A simple transform that takes and arbitrary function for the
    forward and inverse transform.
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, forward, inverse):
        """
        Parameters
        ----------

        forward : callable
            The forward function for the transform.  This function must have
            an inverse and, for best behavior, be monotonic.
            It must have the signature::

               def forward(values: array-like) -> array-like

        inverse : callable
            The inverse of the forward function.  Signature as ``forward``.
        """
        super().__init__()
        if callable(forward) and callable(inverse):
            self._forward = forward
            self._inverse = inverse
        else:
            raise ValueError('arguments to FuncTransform must '
                             'be functions')

    def transform_non_affine(self, values):
        return self._forward(values)

    def inverted(self):
        return FuncTransform(self._inverse, self._forward)


class FuncScale(ScaleBase):
    """
    Provide an arbitrary scale with user-supplied function for the axis.
    """

    name = 'function'

    def __init__(self, axis, functions):
        """
        Parameters
        ----------

        axis: the axis for the scale

        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

               def forward(values: array-like) -> array-like
        """
        forward, inverse = functions
        transform = FuncTransform(forward, inverse)
        self._transform = transform

    def get_transform(self):
        """
        The transform for arbitrary scaling
        """
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to the same defaults as the
        linear scale.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and rcParams['xtick.minor.visible']
            or axis.axis_name == 'y' and rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())


@cbook.deprecated("3.1", alternative="LogTransform")
class LogTransformBase(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, nonpos='clip'):
        Transform.__init__(self)
        self._clip = {"clip": True, "mask": False}[nonpos]

    def transform_non_affine(self, a):
        return LogTransform.transform_non_affine(self, a)

    def __str__(self):
        return "{}({!r})".format(
            type(self).__name__, "clip" if self._clip else "mask")


@cbook.deprecated("3.1", alternative="InvertedLogTransform")
class InvertedLogTransformBase(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def transform_non_affine(self, a):
        return ma.power(self.base, a)

    def __str__(self):
        return "{}()".format(type(self).__name__)


@cbook.deprecated("3.1", alternative="LogTransform")
class Log10Transform(LogTransformBase):
    base = 10.0

    def inverted(self):
        return InvertedLog10Transform()


@cbook.deprecated("3.1", alternative="InvertedLogTransform")
class InvertedLog10Transform(InvertedLogTransformBase):
    base = 10.0

    def inverted(self):
        return Log10Transform()


@cbook.deprecated("3.1", alternative="LogTransform")
class Log2Transform(LogTransformBase):
    base = 2.0

    def inverted(self):
        return InvertedLog2Transform()


@cbook.deprecated("3.1", alternative="InvertedLogTransform")
class InvertedLog2Transform(InvertedLogTransformBase):
    base = 2.0

    def inverted(self):
        return Log2Transform()


@cbook.deprecated("3.1", alternative="LogTransform")
class NaturalLogTransform(LogTransformBase):
    base = np.e

    def inverted(self):
        return InvertedNaturalLogTransform()


@cbook.deprecated("3.1", alternative="InvertedLogTransform")
class InvertedNaturalLogTransform(InvertedLogTransformBase):
    base = np.e

    def inverted(self):
        return NaturalLogTransform()


class LogTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, base, nonpos='clip'):
        Transform.__init__(self)
        self.base = base
        self._clip = {"clip": True, "mask": False}[nonpos]

    def __str__(self):
        return "{}(base={}, nonpos={!r})".format(
            type(self).__name__, self.base, "clip" if self._clip else "mask")

    def transform_non_affine(self, a):
        # Ignore invalid values due to nans being passed to the transform.
        with np.errstate(divide="ignore", invalid="ignore"):
            log = {np.e: np.log, 2: np.log2, 10: np.log10}.get(self.base)
            if log:  # If possible, do everything in a single call to Numpy.
                out = log(a)
            else:
                out = np.log(a)
                out /= np.log(self.base)
            if self._clip:
                # SVG spec says that conforming viewers must support values up
                # to 3.4e38 (C float); however experiments suggest that
                # Inkscape (which uses cairo for rendering) runs into cairo's
                # 24-bit limit (which is apparently shared by Agg).
                # Ghostscript (used for pdf rendering appears to overflow even
                # earlier, with the max value around 2 ** 15 for the tests to
                # pass. On the other hand, in practice, we want to clip beyond
                #     np.log10(np.nextafter(0, 1)) ~ -323
                # so 1000 seems safe.
                out[a <= 0] = -1000
        return out

    def inverted(self):
        return InvertedLogTransform(self.base)


class InvertedLogTransform(InvertedLogTransformBase):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, base):
        Transform.__init__(self)
        self.base = base

    def __str__(self):
        return "{}(base={})".format(type(self).__name__, self.base)

    def transform_non_affine(self, a):
        return ma.power(self.base, a)

    def inverted(self):
        return LogTransform(self.base)


class LogScale(ScaleBase):
    """
    A standard logarithmic scale.  Care is taken to only plot positive values.
    """
    name = 'log'

    # compatibility shim
    LogTransformBase = LogTransformBase
    Log10Transform = Log10Transform
    InvertedLog10Transform = InvertedLog10Transform
    Log2Transform = Log2Transform
    InvertedLog2Transform = InvertedLog2Transform
    NaturalLogTransform = NaturalLogTransform
    InvertedNaturalLogTransform = InvertedNaturalLogTransform
    LogTransform = LogTransform
    InvertedLogTransform = InvertedLogTransform

    def __init__(self, axis, **kwargs):
        """
        *basex*/*basey*:
           The base of the logarithm

        *nonposx*/*nonposy*: {'mask', 'clip'}
          non-positive values in *x* or *y* can be masked as
          invalid, or clipped to a very small positive number

        *subsx*/*subsy*:
           Where to place the subticks between each major tick.
           Should be a sequence of integers.  For example, in a log10
           scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``

           will place 8 logarithmically spaced minor ticks between
           each major tick.
        """
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            subs = kwargs.pop('subsx', None)
            nonpos = kwargs.pop('nonposx', 'clip')
            cbook._check_in_list(['mask', 'clip'], nonposx=nonpos)
        else:
            base = kwargs.pop('basey', 10.0)
            subs = kwargs.pop('subsy', None)
            nonpos = kwargs.pop('nonposy', 'clip')
            cbook._check_in_list(['mask', 'clip'], nonposy=nonpos)

        if len(kwargs):
            raise ValueError(("provided too many kwargs, can only pass "
                              "{'basex', 'subsx', nonposx'} or "
                              "{'basey', 'subsy', nonposy'}.  You passed ") +
                             "{!r}".format(kwargs))

        if base <= 0 or base == 1:
            raise ValueError('The log base cannot be <= 0 or == 1')

        self._transform = self.LogTransform(base, nonpos)
        self.subs = subs

    @property
    def base(self):
        return self._transform.base

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(
            LogFormatterSciNotation(self.base,
                                    labelOnlyBase=(self.subs is not None)))

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        if not np.isfinite(minpos):
            minpos = 1e-300  # This value should rarely if ever
                             # end up with a visible effect.

        return (minpos if vmin <= 0 else vmin,
                minpos if vmax <= 0 else vmax)


class FuncScaleLog(LogScale):
    """
    Provide an arbitrary scale with user-supplied function for the axis and
    then put on a logarithmic axes.
    """

    name = 'functionlog'

    def __init__(self, axis, functions, base=10):
        """
        Parameters
        ----------

        axis: the axis for the scale

        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

                def forward(values: array-like) -> array-like

        base : float
            logarithmic base of the scale (default = 10)

        """
        forward, inverse = functions
        self.subs = None
        self._transform = FuncTransform(forward, inverse) + LogTransform(base)

    @property
    def base(self):
        return self._transform._b.base  # Base of the LogTransform.

    def get_transform(self):
        """
        The transform for arbitrary scaling
        """
        return self._transform


class SymmetricalLogTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, base, linthresh, linscale):
        Transform.__init__(self)
        self.base = base
        self.linthresh = linthresh
        self.linscale = linscale
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))
        self._log_base = np.log(base)

    def transform_non_affine(self, a):
        abs_a = np.abs(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(a) * self.linthresh * (
                self._linscale_adj +
                np.log(abs_a / self.linthresh) / self._log_base)
            inside = abs_a <= self.linthresh
        out[inside] = a[inside] * self._linscale_adj
        return out

    def inverted(self):
        return InvertedSymmetricalLogTransform(self.base, self.linthresh,
                                               self.linscale)


class InvertedSymmetricalLogTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, base, linthresh, linscale):
        Transform.__init__(self)
        symlog = SymmetricalLogTransform(base, linthresh, linscale)
        self.base = base
        self.linthresh = linthresh
        self.invlinthresh = symlog.transform(linthresh)
        self.linscale = linscale
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))

    def transform_non_affine(self, a):
        abs_a = np.abs(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(a) * self.linthresh * (
                np.power(self.base,
                         abs_a / self.linthresh - self._linscale_adj))
            inside = abs_a <= self.invlinthresh
        out[inside] = a[inside] / self._linscale_adj
        return out

    def inverted(self):
        return SymmetricalLogTransform(self.base,
                                       self.linthresh, self.linscale)


class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    Parameters
    ----------
    basex, basey : float
        The base of the logarithm. Defaults to 10.

    linthreshx, linthreshy : float
        Defines the range ``(-x, x)``, within which the plot is linear.
        This avoids having the plot go to infinity around zero. Defaults to 2.

    subsx, subsy : sequence of int
        Where to place the subticks between each major tick.
        For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
        8 logarithmically spaced minor ticks between each major tick.

    linscalex, linscaley : float, optional
        This allows the linear range ``(-linthresh, linthresh)`` to be
        stretched relative to the logarithmic range. Its value is the number of
        decades to use for each half of the linear range. For example, when
        *linscale* == 1.0 (the default), the space used for the positive and
        negative halves of the linear range will be equal to one decade in
        the logarithmic range.
    """
    name = 'symlog'
    # compatibility shim
    SymmetricalLogTransform = SymmetricalLogTransform
    InvertedSymmetricalLogTransform = InvertedSymmetricalLogTransform

    def __init__(self, axis, **kwargs):
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            linthresh = kwargs.pop('linthreshx', 2.0)
            subs = kwargs.pop('subsx', None)
            linscale = kwargs.pop('linscalex', 1.0)
        else:
            base = kwargs.pop('basey', 10.0)
            linthresh = kwargs.pop('linthreshy', 2.0)
            subs = kwargs.pop('subsy', None)
            linscale = kwargs.pop('linscaley', 1.0)

        if base <= 1.0:
            raise ValueError("'basex/basey' must be larger than 1")
        if linthresh <= 0.0:
            raise ValueError("'linthreshx/linthreshy' must be positive")
        if linscale <= 0.0:
            raise ValueError("'linscalex/linthreshy' must be positive")

        self._transform = self.SymmetricalLogTransform(base,
                                                       linthresh,
                                                       linscale)

        self.base = base
        self.linthresh = linthresh
        self.linscale = linscale
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(),
                                                     self.subs))
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`SymmetricalLogTransform` instance.
        """
        return self._transform


class LogitTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, nonpos='mask'):
        Transform.__init__(self)
        self._nonpos = nonpos
        self._clip = {"clip": True, "mask": False}[nonpos]

    def transform_non_affine(self, a):
        """logit transform (base 10), masked or clipped"""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(a / (1 - a))
        if self._clip:  # See LogTransform for choice of clip value.
            out[a <= 0] = -1000
            out[1 <= a] = 1000
        return out

    def inverted(self):
        return LogisticTransform(self._nonpos)

    def __str__(self):
        return "{}({!r})".format(type(self).__name__,
            "clip" if self._clip else "mask")


class LogisticTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, nonpos='mask'):
        Transform.__init__(self)
        self._nonpos = nonpos

    def transform_non_affine(self, a):
        """logistic transform (base 10)"""
        return 1.0 / (1 + 10**(-a))

    def inverted(self):
        return LogitTransform(self._nonpos)

    def __str__(self):
        return "{}({!r})".format(type(self).__name__, self._nonpos)


class LogitScale(ScaleBase):
    """
    Logit scale for data between zero and one, both excluded.

    This scale is similar to a log scale close to zero and to one, and almost
    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
    """
    name = 'logit'

    def __init__(self, axis, nonpos='mask'):
        """
        *nonpos*: {'mask', 'clip'}
          values beyond ]0, 1[ can be masked as invalid, or clipped to a number
          very close to 0 or 1
        """
        cbook._check_in_list(['mask', 'clip'], nonpos=nonpos)
        self._transform = LogitTransform(nonpos)

    def get_transform(self):
        """
        Return a :class:`LogitTransform` instance.
        """
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        # ..., 0.01, 0.1, 0.5, 0.9, 0.99, ...
        axis.set_major_locator(LogitLocator())
        axis.set_major_formatter(LogitFormatter())
        axis.set_minor_locator(LogitLocator(minor=True))
        axis.set_minor_formatter(LogitFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and 1 (excluded).
        """
        if not np.isfinite(minpos):
            minpos = 1e-7    # This value should rarely if ever
                             # end up with a visible effect.
        return (minpos if vmin <= 0 else vmin,
                1 - minpos if vmax >= 1 else vmax)


_scale_mapping = {
    'linear': LinearScale,
    'log':    LogScale,
    'symlog': SymmetricalLogScale,
    'logit':  LogitScale,
    'function': FuncScale,
    'functionlog': FuncScaleLog,
    }


def get_scale_names():
    return sorted(_scale_mapping)


def scale_factory(scale, axis, **kwargs):
    """
    Return a scale class by name.

    Parameters
    ----------
    scale : {%(names)s}
    axis : Axis
    """
    scale = scale.lower()
    if scale not in _scale_mapping:
        raise ValueError("Unknown scale type '%s'" % scale)
    return _scale_mapping[scale](axis, **kwargs)

if scale_factory.__doc__:
    scale_factory.__doc__ = scale_factory.__doc__ % {
        "names": ", ".join(get_scale_names())}


def register_scale(scale_class):
    """
    Register a new kind of scale.

    *scale_class* must be a subclass of :class:`ScaleBase`.
    """
    _scale_mapping[scale_class.name] = scale_class


@cbook.deprecated(
    '3.1', message='get_scale_docs() is considered private API since '
                   '3.1 and will be removed from the public API in 3.3.')
def get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """
    return _get_scale_docs()


def _get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """
    docs = []
    for name, scale_class in _scale_mapping.items():
        docs.extend([
            f"    {name!r}",
            "",
            textwrap.indent(inspect.getdoc(scale_class.__init__), " " * 8),
            ""
        ])
    return "\n".join(docs)


docstring.interpd.update(
    scale=' | '.join([repr(x) for x in get_scale_names()]),
    scale_docs=_get_scale_docs().rstrip(),
    )
