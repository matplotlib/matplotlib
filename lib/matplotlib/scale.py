import numpy as npy
from matplotlib.numerix import npyma as ma
MaskedArray = ma.MaskedArray

from ticker import NullFormatter, ScalarFormatter, LogFormatterMathtext
from ticker import NullLocator, LogLocator, AutoLocator
from transforms import Transform, IdentityTransform

class ScaleBase(object):
    def set_default_locators_and_formatters(self, axis):
        raise NotImplementedError

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax
    
class LinearScale(ScaleBase):
    name = 'linear'
    
    def __init__(self, axis, **kwargs):
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
            
        def transform(self, a):
            a = _mask_non_positives(a * 10.0)
            if isinstance(a, MaskedArray):
                return ma.log10(a)
            return npy.log10(a)
            
        def inverted(self):
            return LogScale.InvertedLog10Transform()

    class InvertedLog10Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
            
        def transform(self, a):
            return ma.power(10.0, a) / 10.0

        def inverted(self):
            return LogScale.Log10Transform()

    class Log2Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
            
        def transform(self, a):
            a = _mask_non_positives(a * 2.0)
            if isinstance(a, MaskedArray):
                return ma.log2(a)
            return npy.log2(a)
            
        def inverted(self):
            return LogScale.InvertedLog2Transform()

    class InvertedLog2Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
            
        def transform(self, a):
            return ma.power(2.0, a) / 2.0

        def inverted(self):
            return LogScale.Log2Transform()

    class NaturalLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform(self, a):
            a = _mask_non_positives(a * npy.e)
            if isinstance(a, MaskedArray):
                return ma.log(a)
            return npy.log(a)
            
        def inverted(self):
            return LogScale.InvertedNaturalLogTransform()

    class InvertedNaturalLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform(self, a):
            return ma.power(npy.e, a) / npy.e

        def inverted(self):
            return LogScale.Log2Transform()
        
    class LogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def __init__(self, base):
            Transform.__init__(self)
            self._base = base
            
        def transform(self, a):
            a = _mask_non_positives(a * self._base)
            if isinstance(a, MaskedArray):
                return ma.log10(a) / npy.log(self._base)
            return npy.log(a) / npy.log(self._base)
            
        def inverted(self):
            return LogScale.InvertedLogTransform(self._base)

    class InvertedLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def __init__(self, base):
            Transform.__init__(self)
            self._base = base

        def transform(self, a):
            return ma.power(self._base, a) / self._base

        def inverted(self):
            return LogScale.LogTransform(self._base)

        
    def __init__(self, axis, **kwargs):
        if axis.axis_name == 'x':
            base = kwargs.pop('basex', 10.0)
            subs = kwargs.pop('subsx', [])
        else:
            base = kwargs.pop('basey', 10.0)
            subs = kwargs.pop('subsy', [])

        if base == 10.0:
            self._transform = self.Log10Transform()
        elif base == 2.0:
            self._transform = self.Log2Transform()
        elif base == npy.e:
            self._transform = self.NaturalLogTransform()
        else:
            self._transform = self.LogTransform(base)

        self._base = base
        self._subs = subs

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogLocator(self._base))
        axis.set_major_formatter(LogFormatterMathtext(self._base))
        axis.set_minor_locator(LogLocator(self._base, self._subs))
        axis.set_minor_formatter(NullFormatter())
            
    def get_transform(self):
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)
    
_scale_mapping = {
    'linear' : LinearScale,
    'log'    : LogScale
    }
def scale_factory(scale, axis, **kwargs):
    scale = scale.lower()
    if scale is None:
        scale = 'linear'

    if not _scale_mapping.has_key(scale):
        raise ValueError("Unknown scale type '%s'" % scale)
    
    return _scale_mapping[scale](axis, **kwargs)

def get_scale_names():
    names = _scale_mapping.keys()
    names.sort()
    return names
