import numpy as npy
from numpy import ma
from numpy.linalg import inv

from ticker import NullFormatter, FixedFormatter, ScalarFormatter, \
    LogFormatter, LogFormatterMathtext
from ticker import NullLocator, FixedLocator, LinearLocator, LogLocator, AutoLocator
from transforms import Affine1DBase, IntervalTransform, Transform, \
    composite_transform_factory, IdentityTransform

class ScaleBase(object):
    def set_default_locators_and_formatters(self, axis):
        raise NotImplementedError
        
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
        
class LogScale(ScaleBase):
    name = 'log'
    
    class Log10Transform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
            
        def transform(self, a):
            return ma.log10(ma.masked_where(a <= 0.0, a * 10.0))
            
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
            return ma.log2(ma.masked_where(a <= 0.0, a * 2.0))
            
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
            return ma.log(ma.masked_where(a <= 0.0, a * npy.e))
            
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
            return ma.log(ma.masked_where(a <= 0.0, a * self._base)) / npy.log(self._base)
            
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
            base = kwargs.pop('basex')
            subs = kwargs.pop('subsx')
        else:
            base = kwargs.pop('basey')
            subs = kwargs.pop('subsy')
            
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
