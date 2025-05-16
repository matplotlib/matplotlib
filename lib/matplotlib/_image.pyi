import numpy

BESSEL: _InterpolationType
BICUBIC: _InterpolationType
BILINEAR: _InterpolationType
BLACKMAN: _InterpolationType
CATROM: _InterpolationType
GAUSSIAN: _InterpolationType
HAMMING: _InterpolationType
HANNING: _InterpolationType
HERMITE: _InterpolationType
KAISER: _InterpolationType
LANCZOS: _InterpolationType
MITCHELL: _InterpolationType
NEAREST: _InterpolationType
QUADRIC: _InterpolationType
SINC: _InterpolationType
SPLINE16: _InterpolationType
SPLINE36: _InterpolationType

class _InterpolationType:
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def resample(input_array: numpy.ndarray, output_array: numpy.ndarray, transform: object, interpolation: _InterpolationType = ..., resample: bool = ..., alpha: float = ..., norm: bool = ..., radius: float = ...) -> None: ...
