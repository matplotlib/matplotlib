import numpy as npy
from numpy import ma
from numpy.linalg import inv

from transforms import Affine1DBase, IntervalTransform, Transform, \
    composite_transform_factory, IdentityTransform

class ScaleBase(object):
    pass

class LinearScale(ScaleBase):
    def get_transform(self):
        return IdentityTransform()
        
class LogScale(ScaleBase):
    class LogTransform(Transform):
        input_dims = 1
        output_dims = 1
        def __init__(self, base):
            Transform.__init__(self)
            self._base = base

        def is_separable(self):
            return True
            
        def transform(self, a):
            if len(a) > 10:
                print "Log Transforming..."
            return ma.log10(ma.masked_where(a <= 0.0, a * 10.0))
            
        def inverted(self):
            return LogScale.InvertedLogTransform(self._base)

    class InvertedLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        def __init__(self, base):
            Transform.__init__(self)
            self._base = base

        def is_separable(self):
            return True
            
        def transform(self, a):
            return ma.power(10.0, a) / 10.0

        def inverted(self):
            return LogScale.LogTransform(self._base)
            
    def __init__(self, base=10):
        self._transform = self.LogTransform(base)

    def get_transform(self):
        return self._transform

    
_scale_mapping = {
    'linear': LinearScale,
    'log': LogScale
    }
def scale_factory(scale, viewLim, direction):
    if scale is None:
        scale = 'linear'
    return _scale_mapping[scale](viewLim, direction)
