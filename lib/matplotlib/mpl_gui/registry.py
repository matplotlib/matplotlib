"""Reproduces the module-level pyplot UX for Figure management."""

from . import FigureRegistry as _FigureRegistry
from ._manage_backend import select_gui_toolkit
from ._manage_interactive import ion, ioff, is_interactive

_fr = _FigureRegistry()

_fr_exports = [
    "figure",
    "subplots",
    "subplot_mosaic",
    "by_label",
    "show",
    "show_all",
    "close",
    "close_all",
]

for k in _fr_exports:
    locals()[k] = getattr(_fr, k)


def get_figlabels():
    return list(_fr.by_label)


def get_fignums():
    return sorted(_fr.by_number)


# if one must.  `from foo import *` is a language miss-feature, but provide
# sensible behavior anyway.
__all__ = _fr_exports + [
    "select_gui_toolkit",
    "ion",
    "ioff",
    "is_interactive",
]
