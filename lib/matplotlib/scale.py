import textwrap
import numpy as np
from numpy import ma
MaskedArray = ma.MaskedArray

from cbook import dedent
from ticker import NullFormatter, ScalarFormatter, LogFormatterMathtext, Formatter
from ticker import NullLocator, LogLocator, AutoLocator, SymmetricalLogLocator, FixedLocator
from ticker import is_decade
from transforms import Transform, IdentityTransform
from matplotlib import docstring

class ScaleBase(object):
    """
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Any subclasses will want to override:

      - :attr:`name`
      - :meth:`get_transform`

    And optionally:
      - :meth:`set_default_locators_and_formatters`
      - :meth:`limit_range_for_scale`
    """
    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` object
        associated with this scale.
        """
        raise NotImplementedError

    def set_default_locators_and_formatters(self, axis):
        """
        Set the :class:`~matplotlib.ticker.Locator` and
        :class:`~matplotlib.ticker.Formatter` objects on the given
        axis to match this scale.
        """
        raise NotImplementedError

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
        pass

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults for
        linear scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        The transform for linear scaling is just the
        :class:`~matplotlib.transforms.IdentityTransform`.
        """
        return IdentityTransform()


def _mask_non_positives(a):
    """
    Return a Numpy masked array where all non-positive values are
    masked.  If there are no non-positive values, the original array
    is returned.
    """
    mask = a <= 0.0
    if mask.any():
        return ma.MaskedArray(a, mask=mask)
    return a

def _clip_non_positives(a):
    a[a <= 0.0] = 1e-300
    return a

class LogScale(ScaleBase):
    """
    A standard logarithmic scale.  Care is taken so non-positive
    values are not plotted.

    For computational efficiency (to push as much as possible to Numpy
    C code in the common cases), this scale provides different
    transforms depending on the base of the logarithm:

       - base 10 (:class:`Log10Transform`)
       - base 2 (:class:`Log2Transform`)
       - base e (:class:`NaturalLogTransform`)
       - arbitrary base (:class:`LogTransform`)
    """

    name = 'log'

    class LogTransformBase(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, nonpos):
            Transform.__init__(self)
            if nonpos == 'mask':
                self._handle_nonpos = _mask_non_positives
            else:
                self._handle_nonpos = _clip_non_positives


    class Log10Transform(LogTransformBase):
        base = 10.0

        def transform(self, a):
            a = self._handle_nonpos(a * 10.0)
            if isinstance(a, MaskedArray):
                return ma.log10(a)
            return np.log10(a)

        def inverted(self):
            return LogScale.InvertedLog10Transform()

    class InvertedLog10Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = 10.0

        def transform(self, a):
            return ma.power(10.0, a) / 10.0

        def inverted(self):
            return LogScale.Log10Transform()

    class Log2Transform(LogTransformBase):
        base = 2.0

        def transform(self, a):
            a = self._handle_nonpos(a * 2.0)
            if isinstance(a, MaskedArray):
                return ma.log(a) / np.log(2)
            return np.log2(a)

        def inverted(self):
            return LogScale.InvertedLog2Transform()

    class InvertedLog2Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = 2.0

        def transform(self, a):
            return ma.power(2.0, a) / 2.0

        def inverted(self):
            return LogScale.Log2Transform()

    class NaturalLogTransform(LogTransformBase):
        base = np.e

        def transform(self, a):
            a = self._handle_nonpos(a * np.e)
            if isinstance(a, MaskedArray):
                return ma.log(a)
            return np.log(a)

        def inverted(self):
            return LogScale.InvertedNaturalLogTransform()

    class InvertedNaturalLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = np.e

        def transform(self, a):
            return ma.power(np.e, a) / np.e

        def inverted(self):
            return LogScale.NaturalLogTransform()

    class LogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, base, nonpos):
            Transform.__init__(self)
            self.base = base
            if nonpos == 'mask':
                self._handle_nonpos = _mask_non_positives
            else:
                self._handle_nonpos = _clip_non_positives

        def transform(self, a):
            a = self._handle_nonpos(a * self.base)
            if isinstance(a, MaskedArray):
                return ma.log(a) / np.log(self.base)
            return np.log(a) / np.log(self.base)

        def inverted(self):
            return LogScale.InvertedLogTransform(self.base)

    class InvertedLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, base):
            Transform.__init__(self)
            self.base = base

        def transform(self, a):
            return ma.power(self.base, a) / self.base

        def inverted(self):
            return LogScale.LogTransform(self.base)


    def __init__(self, axis, **kwargs):
        """
        *basex*/*basey*:
           The base of the logarithm

        *nonposx*/*nonposy*: ['mask' | 'clip' ]
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
            nonpos = kwargs.pop('nonposx', 'mask')
        else:
            base = kwargs.pop('basey', 10.0)
            subs = kwargs.pop('subsy', None)
            nonpos = kwargs.pop('nonposy', 'mask')

        if nonpos not in ['mask', 'clip']:
            raise ValueError("nonposx, nonposy kwarg must be 'mask' or 'clip'")

        if base == 10.0:
            self._transform = self.Log10Transform(nonpos)
        elif base == 2.0:
            self._transform = self.Log2Transform(nonpos)
        elif base == np.e:
            self._transform = self.NaturalLogTransform(nonpos)
        else:
            self._transform = self.LogTransform(base, nonpos)

        self.base = base
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterMathtext(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(NullFormatter())

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
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)


class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).
    """
    name = 'symlog'

    class SymmetricalLogTransform(Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True

            def __init__(self, base, linthresh):
                Transform.__init__(self)
                self.base = base
                self.linthresh = linthresh
                self._log_base = np.log(base)
                self._linadjust = (np.log(linthresh) / self._log_base) / linthresh

            def transform(self, a):
                sign = np.sign(a)
                masked = ma.masked_inside(a, -self.linthresh, self.linthresh, copy=False)
                log = sign * self.linthresh * (1 + ma.log(np.abs(masked) / self.linthresh))
                if masked.mask.any():
                    return ma.where(masked.mask, a, log)
                else:
                    return log

            def inverted(self):
                return SymmetricalLogScale.InvertedSymmetricalLogTransform(self.base, self.linthresh)

    class InvertedSymmetricalLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, base, linthresh):
            Transform.__init__(self)
            self.base = base
            self.linthresh = linthresh
            self._log_base = np.log(base)
            self._log_linthresh = np.log(linthresh) / self._log_base
            self._linadjust = linthresh / (np.log(linthresh) / self._log_base)

        def transform(self, a):
            sign = np.sign(a)
            masked = ma.masked_inside(a, -self.linthresh, self.linthresh, copy=False)
            exp = sign * self.linthresh * ma.exp(sign * masked / self.linthresh - 1)
            if masked.mask.any():
                return ma.where(masked.mask, a, exp)
            else:
                return exp

    def __init__(self, axis, **kwargs):
        """
        *basex*/*basey*:
           The base of the logarithm

        *linthreshx*/*linthreshy*:
          The range (-*x*, *x*) within which the plot is linear (to
          avoid having the plot go to infinity around zero).

        *subsx*/*subsy*:
           Where to place the subticks between each major tick.
           Should be a sequence of integers.  For example, in a log10
           scale: ``[2, 3, 4, 5, 6, 7, 8, 9]``

           will place 8 logarithmically spaced minor ticks between
           each major tick.
        """
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            linthresh = kwargs.pop('linthreshx', 2.0)
            subs = kwargs.pop('subsx', None)
        else:
            base = kwargs.pop('basey', 10.0)
            linthresh = kwargs.pop('linthreshy', 2.0)
            subs = kwargs.pop('subsy', None)

        self._transform = self.SymmetricalLogTransform(base, linthresh)

        assert base > 0.0
        assert linthresh > 0.0

        self.base = base
        self.linthresh = linthresh
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
        axis.set_major_formatter(LogFormatterMathtext(self.base))
        axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(), self.subs))
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`SymmetricalLogTransform` instance.
        """
        return self._transform



_scale_mapping = {
    'linear'            : LinearScale,
    'log'               : LogScale,
    'symlog'            : SymmetricalLogScale
    }
def get_scale_names():
    names = _scale_mapping.keys()
    names.sort()
    return names

def scale_factory(scale, axis, **kwargs):
    """
    Return a scale class by name.

    ACCEPTS: [ %(names)s ]
    """
    scale = scale.lower()
    if scale is None:
        scale = 'linear'

    if scale not in _scale_mapping:
        raise ValueError("Unknown scale type '%s'" % scale)

    return _scale_mapping[scale](axis, **kwargs)
scale_factory.__doc__ = dedent(scale_factory.__doc__) % \
    {'names': " | ".join(get_scale_names())}

def register_scale(scale_class):
    """
    Register a new kind of scale.

    *scale_class* must be a subclass of :class:`ScaleBase`.
    """
    _scale_mapping[scale_class.name] = scale_class

def get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """
    docs = []
    for name in get_scale_names():
        scale_class = _scale_mapping[name]
        docs.append("    '%s'" % name)
        docs.append("")
        class_docs = dedent(scale_class.__init__.__doc__)
        class_docs = "".join(["        %s\n" %
                              x for x in class_docs.split("\n")])
        docs.append(class_docs)
        docs.append("")
    return "\n".join(docs)

docstring.interpd.update(
    scale = ' | '.join([repr(x) for x in get_scale_names()]),
    scale_docs = get_scale_docs().strip(),
    )
