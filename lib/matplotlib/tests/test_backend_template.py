"""
Backend-loading machinery tests, using variations on the template backend.
"""

import sys
from types import SimpleNamespace

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends import backend_template
from matplotlib.backends.backend_template import (
    FigureCanvasTemplate, FigureManagerTemplate)


def test_load_template():
    mpl.use("template")
    assert type(plt.figure().canvas) == FigureCanvasTemplate


def test_load_old_api(monkeypatch):
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    mpl_test_backend.new_figure_manager = (
        lambda num, *args, FigureClass=mpl.figure.Figure, **kwargs:
        FigureManagerTemplate(
            FigureCanvasTemplate(FigureClass(*args, **kwargs)), num))
    monkeypatch.setitem(sys.modules, "mpl_test_backend", mpl_test_backend)
    mpl.use("module://mpl_test_backend")
    assert type(plt.figure().canvas) == FigureCanvasTemplate
    plt.draw_if_interactive()
