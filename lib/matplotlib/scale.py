import numpy as npy
from numpy import ma

from transforms import Affine1D, IntervalTransform, Transform

class ScaleBase(object):
    pass

class LinearScale(ScaleBase):
    def __init__(self, viewLim, direction):
        direction = 'interval' + direction
        self._transform = IntervalTransform(viewLim, direction)

    def get_transform(self):
        return self._transform
        
class LogScale(ScaleBase):
    class LogTransform(Transform):
        input_dims = 1
        output_dims = 1
        def __init__(self, viewLim, direction, base):
            Transform.__init__(self)
            self._base = base
            self._viewLim = viewLim
            self._direction = direction

        def transform(self, a):
            a, affine = self.transform_without_affine(a)
            return affine.transform(a)
            
        def transform_without_affine(self, a):
            # MGDTODO: Support different bases
            base = self._base
            marray = ma.masked_where(a <= 0.0, a)
            marray = npy.log10(marray)
            minimum, maximum = getattr(self._viewLim, self._direction)
            minimum, maximum = npy.log10([minimum, maximum])
            print marray
            print Affine1D.from_values(maximum - minimum, minimum).inverted()
            print minimum, maximum
            return marray, Affine1D.from_values(maximum - minimum, minimum).inverted()
            
        def inverted(self):
            return LogScale.InvertedLogTransform(self._viewLim, self._direction, self._base)

    class InvertedLogTransform(Transform):
        input_dims = 1
        output_dims = 1
        def __init__(self, viewLim, direction, base):
            Transform.__init__(self)
            self._base = base
            self._viewLim = viewLim
            self._direction = direction

        def transform(self, a):
            minimum, maximum = getattr(self._viewLim, self._direction)
            Affine1D.from_values(maximum - minimum, minimum).transform(a)
            return ma.power(10.0, a)

        def inverted(self):
            return LogScale.LogTransform(self._viewLim, self._direction, self._base)
            
    def __init__(self, viewLim, direction, base=10):
        direction = 'interval' + direction
        self._transform = self.LogTransform(viewLim, direction, base)

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
