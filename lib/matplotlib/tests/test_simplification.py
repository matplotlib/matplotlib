import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
from matplotlib import patches, path

from pylab import *
import numpy as np
from matplotlib import patches, path
nan = np.nan
Path = path.Path

# NOTE: All of these tests assume that path.simplify is set to True
# (the default)

@image_comparison(baseline_images=['clipping'])
def test_clipping():
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*pi*t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, s, linewidth=1.0)
    ax.set_ylim((-0.20, -0.28))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('clipping')

@image_comparison(baseline_images=['overflow'])
def test_overflow():
    x = np.array([1.0,2.0,3.0,2.0e5])
    y = np.arange(len(x))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.set_xlim(xmin=2,xmax=6)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('overflow')

@image_comparison(baseline_images=['clipping_diamond'])
def test_diamond():
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    y = np.array([1.0, 0.0, -1.0, 0.0, 1.0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlim(xmin=-0.6, xmax=0.6)
    ax.set_ylim(ymin=-0.6, ymax=0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('clipping_diamond')

def test_noise():
    np.random.seed(0)
    x = np.random.uniform(size=(5000,)) * 50

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)
    ax.set_xticks([])
    ax.set_yticks([])

    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = list(path.iter_segments(simplify=(800, 600)))

    assert len(simplified) == 2662

def test_sine_plus_noise():
    np.random.seed(0)
    x = np.sin(np.linspace(0, np.pi * 2.0, 1000)) + np.random.uniform(size=(1000,)) * 0.01

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)
    ax.set_xticks([])
    ax.set_yticks([])

    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = list(path.iter_segments(simplify=(800, 600)))

    assert len(simplified) == 279

@image_comparison(baseline_images=['simplify_curve'])
def test_simplify_curve():
    pp1 = patches.PathPatch(
        Path([(0, 0), (1, 0), (1, 1), (nan, 1), (0, 0), (2, 0), (2, 2), (0, 0)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
        fc="none")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(pp1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))

    fig.savefig('simplify_curve')

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
