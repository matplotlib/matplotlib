import textwrap
import numpy as np
from numpy import ma
MaskedArray = ma.MaskedArray

from cbook import dedent
from ticker import NullFormatter, ScalarFormatter, LogFormatterMathtext, Formatter
from ticker import NullLocator, LogLocator, AutoLocator, SymmetricalLogLocator, FixedLocator
from transforms import Transform, IdentityTransform

class ScaleBase(object):
    def get_transform(self):
        """
        Return the transform object associated with this scale.
        """
        raise NotImplementedError

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters that go with this scale.
        """
        raise NotImplementedError

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Returns the range vmin, vmax, limited to the domain supported
        by this scale.
        """
        return vmin, vmax

class LinearScale(ScaleBase):
    name = 'linear'

    def __init__(self, axis, **kwargs):
        """
        """
        pass

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        return IdentityTransform()


def _mask_non_positives(a):
    mask = a <= 0.0
    if mask.any():
        return ma.MaskedArray(a, mask=mask)
    return a

class LogScale(ScaleBase):
    name = 'log'

    class Log10Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = 10.0

        def transform(self, a):
            a = _mask_non_positives(a * 10.0)
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

    class Log2Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = 2.0

        def transform(self, a):
            a = _mask_non_positives(a * 2.0)
            if isinstance(a, MaskedArray):
                return ma.log2(a)
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

    class NaturalLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        base = np.e

        def transform(self, a):
            a = _mask_non_positives(a * np.e)
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

        def __init__(self, base):
            Transform.__init__(self)
            self.base = base

        def transform(self, a):
            a = _mask_non_positives(a * self.base)
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

        *subsx*/*subsy*:
           Where to place the subticks between each major tick.
           Should be a sequence of integers.  For example, in a log10
           scale: ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

           will place 10 logarithmically spaced minor ticks between
           each major tick.
        """
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            subs = kwargs.pop('subsx', None)
        else:
            base = kwargs.pop('basey', 10.0)
            subs = kwargs.pop('subsy', None)

        if base == 10.0:
            self._transform = self.Log10Transform()
        elif base == 2.0:
            self._transform = self.Log2Transform()
        elif base == np.e:
            self._transform = self.NaturalLogTransform()
        else:
            self._transform = self.LogTransform(base)

        self.base = base
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterMathtext(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)


class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is
    usually need to have a range around zero that is linear.  The
    parameter "linthresh" allows the user to specify the size of this
    range (-linthresh, linthresh).
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
            sign = np.sign(np.asarray(a))
            masked = ma.masked_inside(a, -self.linthresh, self.linthresh, copy=False)
            log = sign * ma.log(np.abs(masked)) / self._log_base
            if masked.mask.any():
                return np.asarray(ma.where(masked.mask,
                                            a * self._linadjust,
                                            log))
            else:
                return np.asarray(log)

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
            return np.where(a <= self._log_linthresh,
                             np.where(a >= -self._log_linthresh,
                                       a * self._linadjust,
                                       -(np.power(self.base, -a))),
                             np.power(self.base, a))

        def inverted(self):
            return SymmetricalLogScale.SymmetricalLogTransform(self.base)

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
           scale: ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

           will place 10 logarithmically spaced minor ticks between
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

        self.base = base
        self.linthresh = linthresh
        self.subs = subs

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
        axis.set_major_formatter(LogFormatterMathtext(self.base))
        axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(), self.subs))
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        return self._transform



_scale_mapping = {
    'linear'            : LinearScale,
    'log'               : LogScale,
    'symlog'            : SymmetricalLogScale
    }
def scale_factory(scale, axis, **kwargs):
    scale = scale.lower()
    if scale is None:
        scale = 'linear'

    if not _scale_mapping.has_key(scale):
        raise ValueError("Unknown scale type '%s'" % scale)

    return _scale_mapping[scale](axis, **kwargs)

def register_scale(scale_class):
    """
    Register a new kind of scale.
    """
    _scale_mapping[scale_class.name] = scale_class

def get_scale_names():
    names = _scale_mapping.keys()
    names.sort()
    return names

def get_scale_docs():
    docs = []
    for name in get_scale_names():
        scale_class = _scale_mapping[name]
        docs.append("    '%s'" % name)
        docs.append("")
        class_docs = dedent(scale_class.__init__.__doc__)
        class_docs = "".join(["        %s\n" % x for x in class_docs.split("\n")])
        docs.append(class_docs)
        docs.append("")
    return "\n".join(docs)
