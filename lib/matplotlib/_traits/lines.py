"""
matplotlib.lines.Line2D refactored in traitlets
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import warnings

import numpy as np

from . import artist, colors as mcolors, docstring, rcParams
from .artist import Artist, allow_rasterization

# import matplotlib._traits.artist as artist
# from matplotlib._traits.artist import Artist, allow_rasterization

from .cbook import (
    iterable, is_numlike, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, TransformedPath, IdentityTransform

# Imported here for backward compatibility, even though they don't
# really belong.
from numpy import ma
from . import _path
from .markers import (
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)
