"""Some test functions for bivariate interpolation.

Most of these have been yoinked from ACM TOMS 792.
http://netlib.org/toms/792
"""

from __future__ import print_function

import numpy as np
from triangulate import Triangulation


class TestData(dict):
    def __init__(self, *args, **kwds):
        dict.__init__(self, *args, **kwds)
        self.__dict__ = self


class TestDataSet(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

data = TestData(
franke100=TestDataSet(
    x=np.array([0.0227035,  0.0539888,  0.0217008,  0.0175129,  0.0019029,
                -0.0509685,  0.0395408, -0.0487061,  0.0315828, -0.0418785,
                0.1324189,  0.1090271,  0.1254439,  0.093454,   0.0767578,
                0.1451874,  0.0626494,  0.1452734,  0.0958668,  0.0695559,
                0.2645602,  0.2391645,  0.208899,   0.2767329,  0.1714726,
                0.2266781,  0.1909212,  0.1867647,  0.2304634,  0.2426219,
                0.3663168,  0.3857662,  0.3832392,  0.3179087,  0.3466321,
                0.3776591,  0.3873159,  0.3812917,  0.3795364,  0.2803515,
                0.4149771,  0.4277679,  0.420001,   0.4663631,  0.4855658,
                0.4092026,  0.4792578,  0.4812279,  0.3977761,  0.4027321,
                0.5848691,  0.5730076,  0.6063893,  0.5013894,  0.5741311,
                0.6106955,  0.5990105,  0.5380621,  0.6096967,  0.5026188,
                0.6616928,  0.6427836,  0.6396475,  0.6703963,  0.7001181,
                0.633359,   0.6908947,  0.6895638,  0.6718889,  0.6837675,
                0.7736939,  0.7635332,  0.7410424,  0.8258981,  0.7306034,
                0.8086609,  0.8214531,  0.729064,   0.8076643,  0.8170951,
                0.8424572,  0.8684053,  0.8366923,  0.9418461,  0.8478122,
                0.8599583,  0.91757,    0.8596328,  0.9279871,  0.8512805,
                1.044982,   0.9670631,  0.9857884,  0.9676313,  1.0129299,
                0.965704,   1.0019855,  1.0359297,  1.0414677,  0.9471506]),
    y=np.array([-0.0310206,  0.1586742,  0.2576924,  0.3414014,  0.4943596,
                 0.5782854,  0.6993418,  0.7470194,  0.9107649,  0.996289,
                 0.050133,   0.0918555,  0.2592973,  0.3381592,  0.4171125,
                 0.5615563,  0.6552235,  0.7524066,  0.9146523,  0.9632421,
                 0.0292939,  0.0602303,  0.2668783,  0.3696044,  0.4801738,
                 0.5940595,  0.6878797,  0.8185576,  0.9046507,  0.9805412,
                 0.0396955,  0.0684484,  0.2389548,  0.3124129,  0.4902989,
                 0.5199303,  0.6445227,  0.8203789,  0.8938079,  0.9711719,
                -0.0284618,  0.1560965,  0.2262471,  0.3175094,  0.3891417,
                 0.5084949,  0.6324247,  0.7511007,  0.8489712,  0.9978728,
                -0.0271948,  0.127243,   0.2709269,  0.3477728,  0.4259422,
                 0.6084711,  0.6733781,  0.7235242,  0.9242411,  1.0308762,
                 0.0255959,  0.0707835,  0.2008336,  0.3259843,  0.4890704,
                 0.5096324,  0.669788,   0.7759569,  0.9366096,  1.0064516,
                 0.0285374,  0.1021403,  0.1936581,  0.3235775,  0.4714228,
                 0.6091595,  0.6685053,  0.8022808,  0.847679,   1.0512371,
                 0.0380499,  0.0902048,  0.2083092,  0.3318491,  0.4335632,
                 0.5910139,  0.6307383,  0.8144841,  0.904231,   0.969603,
                -0.01209,    0.1334114,  0.2695844,  0.3795281,  0.4396054,
                 0.5044425,  0.6941519,  0.7459923,  0.8682081,  0.9801409])),
franke33=TestDataSet(
    x=np.array([5.00000000e-02,   0.00000000e+00,   0.00000000e+00,
                0.00000000e+00,   1.00000000e-01,   1.00000000e-01,
                1.50000000e-01,   2.00000000e-01,   2.50000000e-01,
                3.00000000e-01,   3.50000000e-01,   5.00000000e-01,
                5.00000000e-01,   5.50000000e-01,   6.00000000e-01,
                6.00000000e-01,   6.00000000e-01,   6.50000000e-01,
                7.00000000e-01,   7.00000000e-01,   7.00000000e-01,
                7.50000000e-01,   7.50000000e-01,   7.50000000e-01,
                8.00000000e-01,   8.00000000e-01,   8.50000000e-01,
                9.00000000e-01,   9.00000000e-01,   9.50000000e-01,
                1.00000000e+00,   1.00000000e+00,   1.00000000e+00]),
    y=np.array([4.50000000e-01,   5.00000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.50000000e-01,   7.50000000e-01,
                3.00000000e-01,   1.00000000e-01,   2.00000000e-01,
                3.50000000e-01,   8.50000000e-01,   0.00000000e+00,
                1.00000000e+00,   9.50000000e-01,   2.50000000e-01,
                6.50000000e-01,   8.50000000e-01,   7.00000000e-01,
                2.00000000e-01,   6.50000000e-01,   9.00000000e-01,
                1.00000000e-01,   3.50000000e-01,   8.50000000e-01,
                4.00000000e-01,   6.50000000e-01,   2.50000000e-01,
                3.50000000e-01,   8.00000000e-01,   9.00000000e-01,
                0.00000000e+00,   5.00000000e-01,   1.00000000e+00])),
lawson25=TestDataSet(
    x=np.array([0.1375,  0.9125,  0.7125,  0.225, -0.05,    0.475,   0.05,
                0.45,    1.0875,  0.5375, -0.0375, 0.1875,  0.7125,  0.85,
                0.7,     0.275,   0.45,    0.8125, 0.45,    1.,      0.5,
                0.1875,  0.5875,  1.05,    0.1]),
    y=np.array([0.975,    0.9875,   0.7625,  0.8375, 0.4125, 0.6375,
               -0.05,     1.0375,   0.55,    0.8,    0.75,   0.575,
                0.55,     0.4375,   0.3125,  0.425,  0.2875, 0.1875,
               -0.0375,   0.2625,   0.4625,  0.2625, 0.125, -0.06125,
                0.1125])),
random100=TestDataSet(
    x=np.array([0.0096326,  0.0216348,  0.029836,   0.0417447,  0.0470462,
                0.0562965,  0.0646857,  0.0740377,  0.0873907,  0.0934832,
                0.1032216,  0.1110176,  0.1181193,  0.1251704,  0.132733,
                0.1439536,  0.1564861,  0.1651043,  0.1786039,  0.1886405,
                0.2016706,  0.2099886,  0.2147003,  0.2204141,  0.2343715,
                0.240966,   0.252774,   0.2570839,  0.2733365,  0.2853833,
                0.2901755,  0.2964854,  0.3019725,  0.3125695,  0.3307163,
                0.3378504,  0.3439061,  0.3529922,  0.3635507,  0.3766172,
                0.3822429,  0.3869838,  0.3973137,  0.4170708,  0.4255588,
                0.4299218,  0.4372839,  0.4705033,  0.4736655,  0.4879299,
                0.494026,   0.5055324,  0.5162593,  0.5219219,  0.5348529,
                0.5483213,  0.5569571,  0.5638611,  0.5784908,  0.586395,
                0.5929148,  0.5987839,  0.6117561,  0.6252296,  0.6331381,
                0.6399048,  0.6488972,  0.6558537,  0.6677405,  0.6814074,
                0.6887812,  0.6940896,  0.7061687,  0.7160957,  0.7317445,
                0.7370798,  0.746203,   0.7566957,  0.7699998,  0.7879347,
                0.7944014,  0.8164468,  0.8192794,  0.8368405,  0.8500993,
                0.8588255,  0.8646496,  0.8792329,  0.8837536,  0.8900077,
                0.8969894,  0.9044917,  0.9083947,  0.9203972,  0.9347906,
                0.9434519,  0.9490328,  0.9569571,  0.9772067,  0.9983493]),
y=np.array([0.3083158,  0.2450434,  0.8613847,  0.0977864,  0.3648355,
            0.7156339,  0.5311312,  0.9755672,  0.1781117,  0.5452797,
            0.1603881,  0.7837139,  0.9982015,  0.6910589,  0.104958,
            0.8184662,  0.7086405,  0.4456593,  0.1178342,  0.3189021,
            0.9668446,  0.7571834,  0.2016598,  0.3232444,  0.4368583,
            0.8907869,  0.064726,   0.5692618,  0.2947027,  0.4332426,
            0.3347464,  0.7436284,  0.1066265,  0.8845357,  0.515873,
            0.9425637,  0.4799701,  0.1783069,  0.114676,   0.8225797,
            0.2270688,  0.4073598,  0.887508,   0.7631616,  0.9972804,
            0.4959884,  0.3410421,  0.249812,   0.6409007,  0.105869,
            0.5411969,  0.0089792,  0.8784268,  0.5515874,  0.4038952,
            0.1654023,  0.2965158,  0.3660356,  0.0366554,  0.950242,
            0.2638101,  0.9277386,  0.5377694,  0.7374676,  0.4674627,
            0.9186109,  0.0416884,  0.1291029,  0.6763676,  0.8444238,
            0.3273328,  0.1893879,  0.0645923,  0.0180147,  0.8904992,
            0.4160648,  0.4688995,  0.2174508,  0.5734231,  0.8853319,
            0.8018436,  0.6388941,  0.8931002,  0.1000558,  0.2789506,
            0.9082948,  0.3259159,  0.8318747,  0.0508513,  0.970845,
            0.5120548,  0.2859716,  0.9581641,  0.6183429,  0.3779934,
            0.4010423,  0.9478657,  0.7425486,  0.8883287,  0.549675])),
uniform9=TestDataSet(
    x=np.array([1.25000000e-01,   0.00000000e+00,   0.00000000e+00,
                0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   1.25000000e-01,
                1.25000000e-01,   1.25000000e-01,   1.25000000e-01,
                1.25000000e-01,   1.25000000e-01,   1.25000000e-01,
                2.50000000e-01,   2.50000000e-01,   2.50000000e-01,
                2.50000000e-01,   2.50000000e-01,   2.50000000e-01,
                2.50000000e-01,   2.50000000e-01,   2.50000000e-01,
                3.75000000e-01,   3.75000000e-01,   3.75000000e-01,
                3.75000000e-01,   3.75000000e-01,   3.75000000e-01,
                3.75000000e-01,   3.75000000e-01,   3.75000000e-01,
                5.00000000e-01,   5.00000000e-01,   5.00000000e-01,
                5.00000000e-01,   5.00000000e-01,   5.00000000e-01,
                5.00000000e-01,   5.00000000e-01,   5.00000000e-01,
                6.25000000e-01,   6.25000000e-01,   6.25000000e-01,
                6.25000000e-01,   6.25000000e-01,   6.25000000e-01,
                6.25000000e-01,   6.25000000e-01,   6.25000000e-01,
                7.50000000e-01,   7.50000000e-01,   7.50000000e-01,
                7.50000000e-01,   7.50000000e-01,   7.50000000e-01,
                7.50000000e-01,   7.50000000e-01,   7.50000000e-01,
                8.75000000e-01,   8.75000000e-01,   8.75000000e-01,
                8.75000000e-01,   8.75000000e-01,   8.75000000e-01,
                8.75000000e-01,   8.75000000e-01,   8.75000000e-01,
                1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
                1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
                1.00000000e+00,   1.00000000e+00,   1.00000000e+00]),
    y=np.array([0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00,
                0.00000000e+00,   1.25000000e-01,   2.50000000e-01,
                3.75000000e-01,   5.00000000e-01,   6.25000000e-01,
                7.50000000e-01,   8.75000000e-01,   1.00000000e+00])),
)


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
    x = x * 9
    y = y * 9
    x1 = x + 1.0
    x2 = x - 2.0
    x4 = x - 4.0
    x7 = x - 7.0
    y1 = x + 1.0
    y2 = y - 2.0
    y3 = y - 3.0
    y7 = y - 7.0
    f = (0.75 * np.exp(-(x2 * x2 + y2 * y2) / 4.0) +
         0.75 * np.exp(-x1 * x1 / 49.0 - y1 / 10.0) +
         0.5 * np.exp(-(x7 * x7 + y3 * y3) / 4.0) -
         0.2 * np.exp(-x4 * x4 - y7 * y7))
    return f
exponential.title = 'Exponential and Some Gaussians'


def cliff(x, y):
    f = np.tanh(9.0 * (y - x) + 1.0) / 9.0
    return f
cliff.title = 'Cliff'


def saddle(x, y):
    f = (1.25 + np.cos(5.4 * y)) / (6.0 + 6.0 * (3 * x - 1.0) ** 2)
    return f
saddle.title = 'Saddle'


def gentle(x, y):
    f = np.exp(-5.0625 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 3.0
    return f
gentle.title = 'Gentle Peak'


def steep(x, y):
    f = np.exp(-20.25 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 3.0
    return f
steep.title = 'Steep Peak'


def sphere(x, y):
    circle = 64 - 81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    f = np.where(circle >= 0, np.sqrt(np.clip(circle, 0, 100)) - 0.5, 0.0)
    return f
sphere.title = 'Sphere'


def trig(x, y):
    f = 2.0 * np.cos(10.0 * x) * np.sin(10.0 * y) + np.sin(10.0 * x * y)
    return f
trig.title = 'Cosines and Sines'


def gauss(x, y):
    x = 5.0 - 10.0 * x
    y = 5.0 - 10.0 * y
    g1 = np.exp(-x * x / 2)
    g2 = np.exp(-y * y / 2)
    f = g1 + 0.75 * g2 * (1 + g1)
    return f
gauss.title = 'Gaussian Peak and Gaussian Ridges'


def cloverleaf(x, y):
    ex = np.exp((10.0 - 20.0 * x) / 3.0)
    ey = np.exp((10.0 - 20.0 * y) / 3.0)
    logitx = 1.0 / (1.0 + ex)
    logity = 1.0 / (1.0 + ey)
    f = (((20.0 / 3.0) ** 3 * ex * ey) ** 2 * (logitx * logity) ** 5 *
        (ex - 2.0 * logitx) * (ey - 2.0 * logity))
    return f
cloverleaf.title = 'Cloverleaf'


def cosine_peak(x, y):
    circle = np.hypot(80 * x - 40.0, 90 * y - 45.)
    f = np.exp(-0.04 * circle) * np.cos(0.15 * circle)
    return f
cosine_peak.title = 'Cosine Peak'

allfuncs = [exponential, cliff, saddle, gentle, steep, sphere, trig, gauss,
            cloverleaf, cosine_peak]


class LinearTester(object):
    name = 'Linear'

    def __init__(self, xrange=(0.0, 1.0), yrange=(0.0, 1.0),
                 nrange=101, npoints=250):
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
        return self.tri.linear_extrapolator(z, bbox=self.xrange + self.yrange)

    def plot(self, func, interp=True, plotter='imshow'):
        import matplotlib as mpl
        from matplotlib import pylab as pl
        if interp:
            lpi = self.interpolator(func)
            z = lpi[self.yrange[0]:self.yrange[1]:complex(0, self.nrange),
                    self.xrange[0]:self.xrange[1]:complex(0, self.nrange)]
        else:
            y, x = np.mgrid[
                self.yrange[0]:self.yrange[1]:complex(0, self.nrange),
                self.xrange[0]:self.xrange[1]:complex(0, self.nrange)]
            z = func(x, y)

        z = np.where(np.isinf(z), 0.0, z)

        extent = (self.xrange[0], self.xrange[1],
            self.yrange[0], self.yrange[1])
        pl.ioff()
        pl.clf()
        pl.hot()  # Some like it hot
        if plotter == 'imshow':
            pl.imshow(np.nan_to_num(z), interpolation='nearest', extent=extent,
                      origin='lower')
        elif plotter == 'contour':
            Y, X = np.ogrid[
                self.yrange[0]:self.yrange[1]:complex(0, self.nrange),
                self.xrange[0]:self.xrange[1]:complex(0, self.nrange)]
            pl.contour(np.ravel(X), np.ravel(Y), z, 20)
        x = self.x
        y = self.y
        lc = mpl.collections.LineCollection(
            np.array([((x[i], y[i]), (x[j], y[j]))
                      for i, j in self.tri.edge_db]),
            colors=[(0, 0, 0, 0.2)])
        ax = pl.gca()
        ax.add_collection(lc)

        if interp:
            title = '%s Interpolant' % self.name
        else:
            title = 'Reference'
        if hasattr(func, 'title'):
            pl.title('%s: %s' % (func.title, title))
        else:
            pl.title(title)

        pl.show()
        pl.ion()


class NNTester(LinearTester):
    name = 'Natural Neighbors'

    def interpolator(self, func):
        z = func(self.x, self.y)
        return self.tri.nn_extrapolator(z, bbox=self.xrange + self.yrange)


def plotallfuncs(allfuncs=allfuncs):
    from matplotlib import pylab as pl
    pl.ioff()
    nnt = NNTester(npoints=1000)
    lpt = LinearTester(npoints=1000)
    for func in allfuncs:
        print(func.title)
        nnt.plot(func, interp=False, plotter='imshow')
        pl.savefig('%s-ref-img.png' % func.func_name)
        nnt.plot(func, interp=True, plotter='imshow')
        pl.savefig('%s-nn-img.png' % func.func_name)
        lpt.plot(func, interp=True, plotter='imshow')
        pl.savefig('%s-lin-img.png' % func.func_name)
        nnt.plot(func, interp=False, plotter='contour')
        pl.savefig('%s-ref-con.png' % func.func_name)
        nnt.plot(func, interp=True, plotter='contour')
        pl.savefig('%s-nn-con.png' % func.func_name)
        lpt.plot(func, interp=True, plotter='contour')
        pl.savefig('%s-lin-con.png' % func.func_name)
    pl.ion()


def plot_dt(tri, colors=None):
    import matplotlib as mpl
    from matplotlib import pylab as pl
    if colors is None:
        colors = [(0, 0, 0, 0.2)]
    lc = mpl.collections.LineCollection(
        np.array([((tri.x[i], tri.y[i]), (tri.x[j], tri.y[j]))
                  for i, j in tri.edge_db]),
        colors=colors)
    ax = pl.gca()
    ax.add_collection(lc)
    pl.draw_if_interactive()


def plot_vo(tri, colors=None):
    import matplotlib as mpl
    from matplotlib import pylab as pl
    if colors is None:
        colors = [(0, 1, 0, 0.2)]
    lc = mpl.collections.LineCollection(np.array(
        [(tri.circumcenters[i], tri.circumcenters[j])
         for i in xrange(len(tri.circumcenters))
         for j in tri.triangle_neighbors[i] if j != -1]),
        colors=colors)
    ax = pl.gca()
    ax.add_collection(lc)
    pl.draw_if_interactive()


def plot_cc(tri, edgecolor=None):
    import matplotlib as mpl
    from matplotlib import pylab as pl
    if edgecolor is None:
        edgecolor = (0, 0, 1, 0.2)
    dxy = (np.array([(tri.x[i], tri.y[i]) for i, j, k in tri.triangle_nodes])
        - tri.circumcenters)
    r = np.hypot(dxy[:, 0], dxy[:, 1])
    ax = pl.gca()
    for i in xrange(len(r)):
        p = mpl.patches.Circle(tri.circumcenters[i], r[i],
                               resolution=100, edgecolor=edgecolor,
                               facecolor=(1, 1, 1, 0), linewidth=0.2)
        ax.add_patch(p)
    pl.draw_if_interactive()


def quality(func, mesh, interpolator='nn', n=33):
    """Compute a quality factor (the quantity r**2 from TOMS792).

    interpolator must be in ('linear', 'nn').
    """
    fz = func(mesh.x, mesh.y)
    tri = Triangulation(mesh.x, mesh.y)
    intp = getattr(tri,
                   interpolator + '_extrapolator')(fz, bbox=(0., 1., 0., 1.))
    Y, X = np.mgrid[0:1:complex(0, n), 0:1:complex(0, n)]
    Z = func(X, Y)
    iz = intp[0:1:complex(0, n), 0:1:complex(0, n)]
    #nans = np.isnan(iz)
    #numgood = n*n - np.sum(np.array(nans.flat, np.int32))
    numgood = n * n

    SE = (Z - iz) ** 2
    SSE = np.sum(SE.flat)
    meanZ = np.sum(Z.flat) / numgood
    SM = (Z - meanZ) ** 2
    SSM = np.sum(SM.flat)

    r2 = 1.0 - SSE / SSM
    print(func.func_name, r2, SSE, SSM, numgood)
    return r2


def allquality(interpolator='nn', allfuncs=allfuncs, data=data, n=33):
    results = {}
    kv = data.items()
    kv.sort()
    for name, mesh in kv:
        reslist = results.setdefault(name, [])
        for func in allfuncs:
            reslist.append(quality(func, mesh, interpolator, n))
    return results


def funky():
    x0 = np.array([0.25, 0.3, 0.5, 0.6, 0.6])
    y0 = np.array([0.2, 0.35, 0.0, 0.25, 0.65])
    tx = 0.46
    ty = 0.23
    t0 = Triangulation(x0, y0)
    t1 = Triangulation(np.hstack((x0, [tx])), np.hstack((y0, [ty])))
    return t0, t1
