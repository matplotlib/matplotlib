# Stub generated from the C++ (pybind11) signatures in ``src/_image_wrapper.cpp``.
# NOT verified with matplotlib's ``stubtest`` (the extension was not importable
# when this stub was written); treat the annotations as best-effort.
from enum import Enum
from typing import cast

import numpy as np

class _InterpolationType(Enum):
    NEAREST = cast(int, ...)
    BILINEAR = cast(int, ...)
    BICUBIC = cast(int, ...)
    SPLINE16 = cast(int, ...)
    SPLINE36 = cast(int, ...)
    HANNING = cast(int, ...)
    HAMMING = cast(int, ...)
    HERMITE = cast(int, ...)
    KAISER = cast(int, ...)
    QUADRIC = cast(int, ...)
    CATROM = cast(int, ...)
    GAUSSIAN = cast(int, ...)
    BESSEL = cast(int, ...)
    MITCHELL = cast(int, ...)
    SINC = cast(int, ...)
    LANCZOS = cast(int, ...)
    BLACKMAN = cast(int, ...)

# ``_InterpolationType`` uses ``.export_values()``, so every member is also
# injected as a module-level constant.
NEAREST: _InterpolationType
BILINEAR: _InterpolationType
BICUBIC: _InterpolationType
SPLINE16: _InterpolationType
SPLINE36: _InterpolationType
HANNING: _InterpolationType
HAMMING: _InterpolationType
HERMITE: _InterpolationType
KAISER: _InterpolationType
QUADRIC: _InterpolationType
CATROM: _InterpolationType
GAUSSIAN: _InterpolationType
BESSEL: _InterpolationType
MITCHELL: _InterpolationType
SINC: _InterpolationType
LANCZOS: _InterpolationType
BLACKMAN: _InterpolationType

def resample(
    input_array: np.ndarray,
    output_array: np.ndarray,
    transform: np.ndarray,
    interpolation: _InterpolationType = ...,
    resample: bool = ...,
    alpha: float = ...,
    norm: bool = ...,
    radius: float = ...,
) -> None: ...
def calculate_rms_and_diff(
    expected_image: np.ndarray, actual_image: np.ndarray
) -> tuple[float, np.ndarray]: ...
