from contextlib import ExitStack
import copy
from numbers import Integral, Number

import numpy as np

import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, colors, ticker,
               transforms)
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
from .widgets import Widget
from .widgets import Button

class New_Button(Button):
    def theme(self, theme):
        self.color = colors("red")
