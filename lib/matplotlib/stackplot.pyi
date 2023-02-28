from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Color

from collections.abc import Iterable
from typing import Literal
from numpy.typing import ArrayLike

def stackplot(
    axes: Axes,
    x: ArrayLike,
    *args: ArrayLike,
    labels: Iterable[str] = ...,
    colors: Iterable[Color] | None = ...,
    baseline: Literal["zero", "sym", "wiggle", "weighted_wiggle"] = ...,
    **kwargs
) -> list[PolyCollection]: ...
