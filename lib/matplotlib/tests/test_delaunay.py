from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange
import warnings

import numpy as np
from matplotlib.testing.decorators import image_comparison, knownfailureif
from matplotlib.cbook import MatplotlibDeprecationWarning

with warnings.catch_warnings():
    # the module is deprecated. The tests should be removed when the module is.
    warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
    from matplotlib.delaunay.triangulate import Triangulation
from matplotlib import pyplot as plt
import matplotlib as mpl

def constant(x, y):
    return np.ones(x.shape, x.dtype)
constant.title = 'Constant'

def xramp(x, y):
    return x
xramp.title = 'X Ramp'

def yramp(x, y):
    return y
yramp.title = 'Y Ramp'

def exponential(x, y):
    x = x*9
    y = y*9
    x1 = x+1.0
    x2 = x-2.0
    x4 = x-4.0
    x7 = x-7.0
    y1 = x+1.0
    y2 = y-2.0
    y3 = y-3.0
    y7 = y-7.0
    f = (0.75 * np.exp(-(x2*x2+y2*y2)/4.0) +
         0.75 * np.exp(-x1*x1/49.0 - y1/10.0) +
         0.5 * np.exp(-(x7*x7 + y3*y3)/4.0) -
         0.2 * np.exp(-x4*x4 -y7*y7))
    return f
exponential.title = 'Exponential and Some Gaussians'

def cliff(x, y):
    f = np.tanh(9.0*(y-x) + 1.0)/9.0
    return f
cliff.title = 'Cliff'

def saddle(x, y):
    f = (1.25 + np.cos(5.4*y))/(6.0 + 6.0*(3*x-1.0)**2)
    return f
saddle.title = 'Saddle'

def gentle(x, y):
    f = np.exp(-5.0625*((x-0.5)**2+(y-0.5)**2))/3.0
    return f
gentle.title = 'Gentle Peak'

def steep(x, y):
    f = np.exp(-20.25*((x-0.5)**2+(y-0.5)**2))/3.0
    return f
steep.title = 'Steep Peak'

def sphere(x, y):
    circle = 64-81*((x-0.5)**2 + (y-0.5)**2)
    f = np.where(circle >= 0, np.sqrt(np.clip(circle,0,100)) - 0.5, 0.0)
    return f
sphere.title = 'Sphere'

def trig(x, y):
    f = 2.0*np.cos(10.0*x)*np.sin(10.0*y) + np.sin(10.0*x*y)
    return f
trig.title = 'Cosines and Sines'

def gauss(x, y):
    x = 5.0-10.0*x
    y = 5.0-10.0*y
    g1 = np.exp(-x*x/2)
    g2 = np.exp(-y*y/2)
    f = g1 + 0.75*g2*(1 + g1)
    return f
gauss.title = 'Gaussian Peak and Gaussian Ridges'

def cloverleaf(x, y):
    ex = np.exp((10.0-20.0*x)/3.0)
    ey = np.exp((10.0-20.0*y)/3.0)
    logitx = 1.0/(1.0+ex)
    logity = 1.0/(1.0+ey)
    f = (((20.0/3.0)**3 * ex*ey)**2 * (logitx*logity)**5 *
        (ex-2.0*logitx)*(ey-2.0*logity))
    return f
cloverleaf.title = 'Cloverleaf'

def cosine_peak(x, y):
    circle = np.hypot(80*x-40.0, 90*y-45.)
    f = np.exp(-0.04*circle) * np.cos(0.15*circle)
    return f
cosine_peak.title = 'Cosine Peak'

allfuncs = [exponential, cliff, saddle, gentle, steep, sphere, trig, gauss, cloverleaf, cosine_peak]


class LinearTester(object):
    name = 'Linear'
    def __init__(self, xrange=(0.0, 1.0), yrange=(0.0, 1.0), nrange=101, npoints=250):
        self.xrange = xrange
        self.yrange = yrange
        self.nrange = nrange
        self.npoints = npoints

        rng = np.random.RandomState(1234567890)
        self.x = rng.uniform(xrange[0], xrange[1], size=npoints)
        self.y = rng.uniform(yrange[0], yrange[1], size=npoints)
        self.tri = Triangulation(self.x, self.y)

    def replace_data(self, dataset):
        self.x = dataset.x
        self.y = dataset.y
        self.tri = Triangulation(self.x, self.y)

    def interpolator(self, func):
        z = func(self.x, self.y)
        return self.tri.linear_extrapolator(z, bbox=self.xrange+self.yrange)

    def plot(self, func, interp=True, plotter='imshow'):
        if interp:
            lpi = self.interpolator(func)
            z = lpi[self.yrange[0]:self.yrange[1]:complex(0,self.nrange),
                    self.xrange[0]:self.xrange[1]:complex(0,self.nrange)]
        else:
            y, x = np.mgrid[self.yrange[0]:self.yrange[1]:complex(0,self.nrange),
                            self.xrange[0]:self.xrange[1]:complex(0,self.nrange)]
            z = func(x, y)

        z = np.where(np.isinf(z), 0.0, z)

        extent = (self.xrange[0], self.xrange[1],
            self.yrange[0], self.yrange[1])
        fig = plt.figure()
        plt.hot() # Some like it hot
        if plotter == 'imshow':
            plt.imshow(np.nan_to_num(z), interpolation='nearest', extent=extent, origin='lower')
        elif plotter == 'contour':
            Y, X = np.ogrid[self.yrange[0]:self.yrange[1]:complex(0,self.nrange),
                self.xrange[0]:self.xrange[1]:complex(0,self.nrange)]
            plt.contour(np.ravel(X), np.ravel(Y), z, 20)
        x = self.x
        y = self.y
        lc = mpl.collections.LineCollection(np.array([((x[i], y[i]), (x[j], y[j]))
            for i, j in self.tri.edge_db]), colors=[(0,0,0,0.2)])
        ax = plt.gca()
        ax.add_collection(lc)

        if interp:
            title = '%s Interpolant' % self.name
        else:
            title = 'Reference'
        if hasattr(func, 'title'):
            plt.title('%s: %s' % (func.title, title))
        else:
            plt.title(title)

class NNTester(LinearTester):
    name = 'Natural Neighbors'
    def interpolator(self, func):
        z = func(self.x, self.y)
        return self.tri.nn_extrapolator(z, bbox=self.xrange+self.yrange)

def make_all_2d_testfuncs(allfuncs=allfuncs):
    def make_test(func):
        filenames = [
            '%s-%s' % (func.__name__, x) for x in
            ['ref-img', 'nn-img', 'lin-img', 'ref-con', 'nn-con', 'lin-con']]

        # We only generate PNGs to save disk space -- we just assume
        # that any backend differences are caught by other tests.
        @image_comparison(filenames, extensions=['png'],
                          freetype_version=('2.4.5', '2.4.9'),
                          remove_text=True)
        def reference_test():
            nnt.plot(func, interp=False, plotter='imshow')
            nnt.plot(func, interp=True, plotter='imshow')
            lpt.plot(func, interp=True, plotter='imshow')
            nnt.plot(func, interp=False, plotter='contour')
            nnt.plot(func, interp=True, plotter='contour')
            lpt.plot(func, interp=True, plotter='contour')

        tester = reference_test
        tester.__name__ = str('test_%s' % func.__name__)
        return tester

    nnt = NNTester(npoints=1000)
    lpt = LinearTester(npoints=1000)
    for func in allfuncs:
        globals()['test_%s' % func.__name__] = make_test(func)

make_all_2d_testfuncs()

# 1d and 0d grid tests

ref_interpolator = Triangulation([0,10,10,0],
                                 [0,0,10,10]).linear_interpolator([1,10,5,2.0])

def test_1d_grid():
    res = ref_interpolator[3:6:2j,1:1:1j]
    assert np.allclose(res, [[1.6],[1.9]], rtol=0)

def test_0d_grid():
    res = ref_interpolator[3:3:1j,1:1:1j]
    assert np.allclose(res, [[1.6]], rtol=0)

@image_comparison(baseline_images=['delaunay-1d-interp'], extensions=['png'])
def test_1d_plots():
    x_range = slice(0.25,9.75,20j)
    x = np.mgrid[x_range]
    ax = plt.gca()
    for y in xrange(2,10,2):
        plt.plot(x, ref_interpolator[x_range,y:y:1j])
    ax.set_xticks([])
    ax.set_yticks([])
