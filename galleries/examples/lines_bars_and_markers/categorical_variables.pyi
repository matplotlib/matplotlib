from __future__ import annotations
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

data: dict[str, int]
names: list[str]
values: list[int]
fig: Figure
axs: NDArray[Axes]
cat: list[str]
dog: list[str]
activity: list[str]
ax: Axes
