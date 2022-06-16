"""
Backend-loading machinery tests, using variations on the template backend.
"""

import sys
from types import SimpleNamespace

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends import backend_template


def test_load_template():
    mpl.use("template")
    assert type(plt.figure().canvas) == backend_template.FigureCanvasTemplate


def test_new_manager(monkeypatch):
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    del mpl_test_backend.new_figure_manager
    monkeypatch.setitem(sys.modules, "mpl_test_backend", mpl_test_backend)
    mpl.use("module://mpl_test_backend")
    assert type(plt.figure().canvas) == backend_template.FigureCanvasTemplate
