from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import tempfile

from numpy.testing import assert_allclose, assert_array_equal
import numpy.ma.testutils as matest
import numpy as np
from nose.tools import (assert_equal, assert_almost_equal, assert_not_equal,
                        assert_true, assert_raises)

import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
from matplotlib.testing.decorators import knownfailureif, CleanupTestCase


try:
    from mpl_toolkits.natgrid import _natgrid
    HAS_NATGRID = True
except ImportError:
    HAS_NATGRID = False


class general_testcase(CleanupTestCase):
    def test_colinear_pca(self):
        a = mlab.PCA._get_colinear()
        pca = mlab.PCA(a)

        assert_allclose(pca.fracs[2:], 0., atol=1e-8)
        assert_allclose(pca.Y[:, 2:], 0., atol=1e-8)

    def test_prctile(self):
        # test odd lengths
        x = [1, 2, 3]
        assert_equal(mlab.prctile(x, 50), np.median(x))

        # test even lengths
        x = [1, 2, 3, 4]
        assert_equal(mlab.prctile(x, 50), np.median(x))

        # derived from email sent by jason-sage to MPL-user on 20090914
        ob1 = [1, 1, 2, 2, 1, 2, 4, 3, 2, 2, 2, 3,
               4, 5, 6, 7, 8, 9, 7, 6, 4, 5, 5]
        p = [0, 75, 100]
        expected = [1, 5.5, 9]

        # test vectorized
        actual = mlab.prctile(ob1, p)
        assert_allclose(expected, actual)

        # test scalar
        for pi, expectedi in zip(p, expected):
            actuali = mlab.prctile(ob1, pi)
            assert_allclose(expectedi, actuali)

    def test_norm(self):
        np.random.seed(0)
        N = 1000
        x = np.random.standard_normal(N)
        targ = np.linalg.norm(x)
        res = mlab._norm(x)
        assert_almost_equal(targ, res)


class spacing_testcase(CleanupTestCase):
    def test_logspace_tens(self):
        xmin = .01
        xmax = 1000.
        N = 6
        res = mlab.logspace(xmin, xmax, N)
        targ = np.logspace(np.log10(xmin), np.log10(xmax), N)
        assert_allclose(targ, res)

    def test_logspace_primes(self):
        xmin = .03
        xmax = 1313.
        N = 7
        res = mlab.logspace(xmin, xmax, N)
        targ = np.logspace(np.log10(xmin), np.log10(xmax), N)
        assert_allclose(targ, res)

    def test_logspace_none(self):
        xmin = .03
        xmax = 1313.
        N = 0
        res = mlab.logspace(xmin, xmax, N)
        targ = np.logspace(np.log10(xmin), np.log10(xmax), N)
        assert_array_equal(targ, res)
        assert_equal(res.size, 0)

    def test_logspace_single(self):
        xmin = .03
        xmax = 1313.
        N = 1
        res = mlab.logspace(xmin, xmax, N)
        targ = np.logspace(np.log10(xmin), np.log10(xmax), N)
        assert_array_equal(targ, res)
        assert_equal(res.size, 1)


class stride_testcase(CleanupTestCase):
    def get_base(self, x):
        y = x
        while y.base is not None:
            y = y.base
        return y

    def calc_window_target(self, x, NFFT, noverlap=0):
        '''This is an adaptation of the original window extraction
        algorithm.  This is here to test to make sure the new implementation
        has the same result'''
        step = NFFT - noverlap
        ind = np.arange(0, len(x) - NFFT + 1, step)
        n = len(ind)
        result = np.zeros((NFFT, n))

        # do the ffts of the slices
        for i in range(n):
            result[:, i] = x[ind[i]:ind[i]+NFFT]
        return result

    def test_stride_windows_2D_ValueError(self):
        x = np.arange(10)[np.newaxis]
        assert_raises(ValueError, mlab.stride_windows, x, 5)

    def test_stride_windows_0D_ValueError(self):
        x = np.array(0)
        assert_raises(ValueError, mlab.stride_windows, x, 5)

    def test_stride_windows_noverlap_gt_n_ValueError(self):
        x = np.arange(10)
        assert_raises(ValueError, mlab.stride_windows, x, 2, 3)

    def test_stride_windows_noverlap_eq_n_ValueError(self):
        x = np.arange(10)
        assert_raises(ValueError, mlab.stride_windows, x, 2, 2)

    def test_stride_windows_n_gt_lenx_ValueError(self):
        x = np.arange(10)
        assert_raises(ValueError, mlab.stride_windows, x, 11)

    def test_stride_windows_n_lt_1_ValueError(self):
        x = np.arange(10)
        assert_raises(ValueError, mlab.stride_windows, x, 0)

    def test_stride_repeat_2D_ValueError(self):
        x = np.arange(10)[np.newaxis]
        assert_raises(ValueError, mlab.stride_repeat, x, 5)

    def test_stride_repeat_axis_lt_0_ValueError(self):
        x = np.array(0)
        assert_raises(ValueError, mlab.stride_repeat, x, 5, axis=-1)

    def test_stride_repeat_axis_gt_1_ValueError(self):
        x = np.array(0)
        assert_raises(ValueError, mlab.stride_repeat, x, 5, axis=2)

    def test_stride_repeat_n_lt_1_ValueError(self):
        x = np.arange(10)
        assert_raises(ValueError, mlab.stride_repeat, x, 0)

    def test_stride_repeat_n1_axis0(self):
        x = np.arange(10)
        y = mlab.stride_repeat(x, 1)
        assert_equal((1, ) + x.shape, y.shape)
        assert_array_equal(x, y.flat)
        assert_true(self.get_base(y) is x)

    def test_stride_repeat_n1_axis1(self):
        x = np.arange(10)
        y = mlab.stride_repeat(x, 1, axis=1)
        assert_equal(x.shape + (1, ), y.shape)
        assert_array_equal(x, y.flat)
        assert_true(self.get_base(y) is x)

    def test_stride_repeat_n5_axis0(self):
        x = np.arange(10)
        y = mlab.stride_repeat(x, 5)
        yr = np.repeat(x[np.newaxis], 5, axis=0)
        assert_equal(yr.shape, y.shape)
        assert_array_equal(yr, y)
        assert_equal((5, ) + x.shape, y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_repeat_n5_axis1(self):
        x = np.arange(10)
        y = mlab.stride_repeat(x, 5, axis=1)
        yr = np.repeat(x[np.newaxis], 5, axis=0).T
        assert_equal(yr.shape, y.shape)
        assert_array_equal(yr, y)
        assert_equal(x.shape + (5, ), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n1_noverlap0_axis0(self):
        x = np.arange(10)
        y = mlab.stride_windows(x, 1)
        yt = self.calc_window_target(x, 1)
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((1, ) + x.shape, y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n1_noverlap0_axis1(self):
        x = np.arange(10)
        y = mlab.stride_windows(x, 1, axis=1)
        yt = self.calc_window_target(x, 1).T
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal(x.shape + (1, ), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n5_noverlap0_axis0(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 5)
        yt = self.calc_window_target(x, 5)
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((5, 20), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n5_noverlap0_axis1(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 5, axis=1)
        yt = self.calc_window_target(x, 5).T
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((20, 5), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n15_noverlap2_axis0(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 15, 2)
        yt = self.calc_window_target(x, 15, 2)
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((15, 7), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n15_noverlap2_axis1(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 15, 2, axis=1)
        yt = self.calc_window_target(x, 15, 2).T
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((7, 15), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n13_noverlapn3_axis0(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 13, -3)
        yt = self.calc_window_target(x, 13, -3)
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((13, 6), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n13_noverlapn3_axis1(self):
        x = np.arange(100)
        y = mlab.stride_windows(x, 13, -3, axis=1)
        yt = self.calc_window_target(x, 13, -3).T
        assert_equal(yt.shape, y.shape)
        assert_array_equal(yt, y)
        assert_equal((6, 13), y.shape)
        assert_true(self.get_base(y) is x)

    def test_stride_windows_n32_noverlap0_axis0_unflatten(self):
        n = 32
        x = np.arange(n)[np.newaxis]
        x1 = np.tile(x, (21, 1))
        x2 = x1.flatten()
        y = mlab.stride_windows(x2, n)
        assert_equal(y.shape, x1.T.shape)
        assert_array_equal(y, x1.T)

    def test_stride_windows_n32_noverlap0_axis1_unflatten(self):
        n = 32
        x = np.arange(n)[np.newaxis]
        x1 = np.tile(x, (21, 1))
        x2 = x1.flatten()
        y = mlab.stride_windows(x2, n, axis=1)
        assert_equal(y.shape, x1.shape)
        assert_array_equal(y, x1)

    def test_stride_ensure_integer_type(self):
        N = 100
        x = np.empty(N + 20, dtype='>f4')
        x.fill(np.NaN)
        y = x[10:-10]
        y.fill(0.3)
        # previous to #3845 lead to corrupt access
        y_strided = mlab.stride_windows(y, n=33, noverlap=0.6)
        assert_array_equal(y_strided, 0.3)
        # previous to #3845 lead to corrupt access
        y_strided = mlab.stride_windows(y, n=33.3, noverlap=0)
        assert_array_equal(y_strided, 0.3)
        # even previous to #3845 could not find any problematic
        # configuration however, let's be sure it's not accidentally
        # introduced
        y_strided = mlab.stride_repeat(y, n=33.815)
        assert_array_equal(y_strided, 0.3)


class csv_testcase(CleanupTestCase):
    def setUp(self):
        if six.PY3:
            self.fd = tempfile.TemporaryFile(suffix='csv', mode="w+",
                                             newline='')
        else:
            self.fd = tempfile.TemporaryFile(suffix='csv', mode="wb+")

    def tearDown(self):
        self.fd.close()

    def test_recarray_csv_roundtrip(self):
        expected = np.recarray((99,),
                               [(str('x'), np.float),
                                (str('y'), np.float),
                                (str('t'), np.float)])
        # initialising all values: uninitialised memory sometimes produces
        # floats that do not round-trip to string and back.
        expected['x'][:] = np.linspace(-1e9, -1, 99)
        expected['y'][:] = np.linspace(1, 1e9, 99)
        expected['t'][:] = np.linspace(0, 0.01, 99)

        mlab.rec2csv(expected, self.fd)
        self.fd.seek(0)
        actual = mlab.csv2rec(self.fd)

        assert_allclose(expected['x'], actual['x'])
        assert_allclose(expected['y'], actual['y'])
        assert_allclose(expected['t'], actual['t'])

    def test_rec2csv_bad_shape_ValueError(self):
        bad = np.recarray((99, 4), [(str('x'), np.float),
                                    (str('y'), np.float)])

        # the bad recarray should trigger a ValueError for having ndim > 1.
        assert_raises(ValueError, mlab.rec2csv, bad, self.fd)


class window_testcase(CleanupTestCase):
    def setUp(self):
        np.random.seed(0)
        self.n = 1000
        self.x = np.arange(0., self.n)

        self.sig_rand = np.random.standard_normal(self.n) + 100.
        self.sig_ones = np.ones_like(self.x)
        self.sig_slope = np.linspace(-10., 90., self.n)

    def check_window_apply_repeat(self, x, window, NFFT, noverlap):
        '''This is an adaptation of the original window application
        algorithm.  This is here to test to make sure the new implementation
        has the same result'''
        step = NFFT - noverlap
        ind = np.arange(0, len(x) - NFFT + 1, step)
        n = len(ind)
        result = np.zeros((NFFT, n))

        if cbook.iterable(window):
            windowVals = window
        else:
            windowVals = window(np.ones((NFFT,), x.dtype))

        # do the ffts of the slices
        for i in range(n):
            result[:, i] = windowVals * x[ind[i]:ind[i]+NFFT]
        return result

    def test_window_none_rand(self):
        res = mlab.window_none(self.sig_ones)
        assert_array_equal(res, self.sig_ones)

    def test_window_none_ones(self):
        res = mlab.window_none(self.sig_rand)
        assert_array_equal(res, self.sig_rand)

    def test_window_hanning_rand(self):
        targ = np.hanning(len(self.sig_rand)) * self.sig_rand
        res = mlab.window_hanning(self.sig_rand)

        assert_allclose(targ, res, atol=1e-06)

    def test_window_hanning_ones(self):
        targ = np.hanning(len(self.sig_ones))
        res = mlab.window_hanning(self.sig_ones)

        assert_allclose(targ, res, atol=1e-06)

    def test_apply_window_1D_axis1_ValueError(self):
        x = self.sig_rand
        window = mlab.window_hanning
        assert_raises(ValueError, mlab.apply_window, x, window, axis=1,
                      return_window=False)

    def test_apply_window_1D_els_wrongsize_ValueError(self):
        x = self.sig_rand
        window = mlab.window_hanning(np.ones(x.shape[0]-1))
        assert_raises(ValueError, mlab.apply_window, x, window)

    def test_apply_window_0D_ValueError(self):
        x = np.array(0)
        window = mlab.window_hanning
        assert_raises(ValueError, mlab.apply_window, x, window, axis=1,
                      return_window=False)

    def test_apply_window_3D_ValueError(self):
        x = self.sig_rand[np.newaxis][np.newaxis]
        window = mlab.window_hanning
        assert_raises(ValueError, mlab.apply_window, x, window, axis=1,
                      return_window=False)

    def test_apply_window_hanning_1D(self):
        x = self.sig_rand
        window = mlab.window_hanning
        window1 = mlab.window_hanning(np.ones(x.shape[0]))
        y, window2 = mlab.apply_window(x, window, return_window=True)
        yt = window(x)
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)
        assert_array_equal(window1, window2)

    def test_apply_window_hanning_1D_axis0(self):
        x = self.sig_rand
        window = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=0, return_window=False)
        yt = window(x)
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_els_1D_axis0(self):
        x = self.sig_rand
        window = mlab.window_hanning(np.ones(x.shape[0]))
        window1 = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=0, return_window=False)
        yt = window1(x)
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_2D_axis0(self):
        x = np.random.standard_normal([1000, 10]) + 100.
        window = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=0, return_window=False)
        yt = np.zeros_like(x)
        for i in range(x.shape[1]):
            yt[:, i] = window(x[:, i])
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_els1_2D_axis0(self):
        x = np.random.standard_normal([1000, 10]) + 100.
        window = mlab.window_hanning(np.ones(x.shape[0]))
        window1 = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=0, return_window=False)
        yt = np.zeros_like(x)
        for i in range(x.shape[1]):
            yt[:, i] = window1(x[:, i])
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_els2_2D_axis0(self):
        x = np.random.standard_normal([1000, 10]) + 100.
        window = mlab.window_hanning
        window1 = mlab.window_hanning(np.ones(x.shape[0]))
        y, window2 = mlab.apply_window(x, window, axis=0, return_window=True)
        yt = np.zeros_like(x)
        for i in range(x.shape[1]):
            yt[:, i] = window1*x[:, i]
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)
        assert_array_equal(window1, window2)

    def test_apply_window_hanning_els3_2D_axis0(self):
        x = np.random.standard_normal([1000, 10]) + 100.
        window = mlab.window_hanning
        window1 = mlab.window_hanning(np.ones(x.shape[0]))
        y, window2 = mlab.apply_window(x, window, axis=0, return_window=True)
        yt = mlab.apply_window(x, window1, axis=0, return_window=False)
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)
        assert_array_equal(window1, window2)

    def test_apply_window_hanning_2D_axis1(self):
        x = np.random.standard_normal([10, 1000]) + 100.
        window = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=1, return_window=False)
        yt = np.zeros_like(x)
        for i in range(x.shape[0]):
            yt[i, :] = window(x[i, :])
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_2D__els1_axis1(self):
        x = np.random.standard_normal([10, 1000]) + 100.
        window = mlab.window_hanning(np.ones(x.shape[1]))
        window1 = mlab.window_hanning
        y = mlab.apply_window(x, window, axis=1, return_window=False)
        yt = np.zeros_like(x)
        for i in range(x.shape[0]):
            yt[i, :] = window1(x[i, :])
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_2D_els2_axis1(self):
        x = np.random.standard_normal([10, 1000]) + 100.
        window = mlab.window_hanning
        window1 = mlab.window_hanning(np.ones(x.shape[1]))
        y, window2 = mlab.apply_window(x, window, axis=1, return_window=True)
        yt = np.zeros_like(x)
        for i in range(x.shape[0]):
            yt[i, :] = window1 * x[i, :]
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)
        assert_array_equal(window1, window2)

    def test_apply_window_hanning_2D_els3_axis1(self):
        x = np.random.standard_normal([10, 1000]) + 100.
        window = mlab.window_hanning
        window1 = mlab.window_hanning(np.ones(x.shape[1]))
        y = mlab.apply_window(x, window, axis=1, return_window=False)
        yt = mlab.apply_window(x, window1, axis=1, return_window=False)
        assert_equal(yt.shape, y.shape)
        assert_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_stride_windows_hanning_2D_n13_noverlapn3_axis0(self):
        x = self.sig_rand
        window = mlab.window_hanning
        yi = mlab.stride_windows(x, n=13, noverlap=2, axis=0)
        y = mlab.apply_window(yi, window, axis=0, return_window=False)
        yt = self.check_window_apply_repeat(x, window, 13, 2)
        assert_equal(yt.shape, y.shape)
        assert_not_equal(x.shape, y.shape)
        assert_allclose(yt, y, atol=1e-06)

    def test_apply_window_hanning_2D_stack_axis1(self):
        ydata = np.arange(32)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ycontrol1 = mlab.apply_window(ydata1, mlab.window_hanning)
        ycontrol2 = mlab.window_hanning(ydata2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        result = mlab.apply_window(ydata, mlab.window_hanning, axis=1,
                                   return_window=False)
        assert_allclose(ycontrol, result, atol=1e-08)

    def test_apply_window_hanning_2D_stack_windows_axis1(self):
        ydata = np.arange(32)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ycontrol1 = mlab.apply_window(ydata1, mlab.window_hanning)
        ycontrol2 = mlab.window_hanning(ydata2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        result = mlab.apply_window(ydata, mlab.window_hanning, axis=1,
                                   return_window=False)
        assert_allclose(ycontrol, result, atol=1e-08)

    def test_apply_window_hanning_2D_stack_windows_axis1_unflatten(self):
        n = 32
        ydata = np.arange(n)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ycontrol1 = mlab.apply_window(ydata1, mlab.window_hanning)
        ycontrol2 = mlab.window_hanning(ydata2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        ydata = ydata.flatten()
        ydata1 = mlab.stride_windows(ydata, 32, noverlap=0, axis=0)
        result = mlab.apply_window(ydata1, mlab.window_hanning, axis=0,
                                   return_window=False)
        assert_allclose(ycontrol.T, result, atol=1e-08)


class detrend_testcase(CleanupTestCase):
    def setUp(self):
        np.random.seed(0)
        n = 1000
        x = np.linspace(0., 100, n)

        self.sig_zeros = np.zeros(n)

        self.sig_off = self.sig_zeros + 100.
        self.sig_slope = np.linspace(-10., 90., n)

        self.sig_slope_mean = x - x.mean()

        sig_rand = np.random.standard_normal(n)
        sig_sin = np.sin(x*2*np.pi/(n/100))

        sig_rand -= sig_rand.mean()
        sig_sin -= sig_sin.mean()

        self.sig_base = sig_rand + sig_sin

        self.atol = 1e-08

    def test_detrend_none_0D_zeros(self):
        input = 0.
        targ = input
        res = mlab.detrend_none(input)
        assert_equal(input, targ)

    def test_detrend_none_0D_zeros_axis1(self):
        input = 0.
        targ = input
        res = mlab.detrend_none(input, axis=1)
        assert_equal(input, targ)

    def test_detrend_str_none_0D_zeros(self):
        input = 0.
        targ = input
        res = mlab.detrend(input, key='none')
        assert_equal(input, targ)

    def test_detrend_detrend_none_0D_zeros(self):
        input = 0.
        targ = input
        res = mlab.detrend(input, key=mlab.detrend_none)
        assert_equal(input, targ)

    def test_detrend_none_0D_off(self):
        input = 5.5
        targ = input
        res = mlab.detrend_none(input)
        assert_equal(input, targ)

    def test_detrend_none_1D_off(self):
        input = self.sig_off
        targ = input
        res = mlab.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_slope(self):
        input = self.sig_slope
        targ = input
        res = mlab.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_base(self):
        input = self.sig_base
        targ = input
        res = mlab.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_base_slope_off_list(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = input.tolist()
        res = mlab.detrend_none(input.tolist())
        assert_equal(res, targ)

    def test_detrend_none_2D(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)
        targ = input
        res = mlab.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_2D_T(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)
        targ = input
        res = mlab.detrend_none(input.T)
        assert_array_equal(res.T, targ)

    def test_detrend_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = mlab.detrend_mean(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = mlab.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = mlab.detrend(input, key=mlab.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend_mean(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend(input, key=mlab.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_1D_zeros(self):
        input = self.sig_zeros
        targ = self.sig_zeros
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base(self):
        input = self.sig_base
        targ = self.sig_base
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_off(self):
        input = self.sig_base + self.sig_off
        targ = self.sig_base
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_slope(self):
        input = self.sig_base + self.sig_slope
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_slope_off(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_axis0(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.detrend_mean(input, axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_list(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.detrend_mean(input.tolist())
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_list_axis0(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.detrend_mean(input.tolist(), axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_demean_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.demean(input, axis=None)
        assert_almost_equal(res, targ)

    def test_demean_1D_base_slope_off(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.demean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_demean_1D_base_slope_off_axis0(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.demean(input, axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_demean_1D_base_slope_off_list(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = mlab.demean(input.tolist())
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2D_default(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=None)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_none_T(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend_mean(input, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axism1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=-1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=None)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_none_T(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend_mean(input, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axism1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend_mean(input, axis=-1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_2D_default(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend(input, axis=None)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_str_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend(input, key='mean', axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_str_constant_2D_none_T(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T
        targ = np.vstack(arrt)
        res = mlab.detrend(input, key='constant', axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_str_default_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend(input, key='default', axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_detrend_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend(input, key=mlab.detrend_mean, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_demean_2D_default(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.demean(input)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_demean_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.demean(input, axis=None)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_demean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.demean(input, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_demean_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.demean(input, axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_demean_2D_axism1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.demean(input, axis=-1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_bad_key_str_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.detrend, input, key='spam')

    def test_detrend_bad_key_var_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.detrend, input, key=5)

    def test_detrend_mean_0D_d0_ValueError(self):
        input = 5.5
        assert_raises(ValueError, mlab.detrend_mean, input, axis=0)

    def test_detrend_0D_d0_ValueError(self):
        input = 5.5
        assert_raises(ValueError, mlab.detrend, input, axis=0)

    def test_detrend_mean_1D_d1_ValueError(self):
        input = self.sig_slope
        assert_raises(ValueError, mlab.detrend_mean, input, axis=1)

    def test_detrend_1D_d1_ValueError(self):
        input = self.sig_slope
        assert_raises(ValueError, mlab.detrend, input, axis=1)

    def test_demean_1D_d1_ValueError(self):
        input = self.sig_slope
        assert_raises(ValueError, mlab.demean, input, axis=1)

    def test_detrend_mean_2D_d2_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.detrend_mean, input, axis=2)

    def test_detrend_2D_d2_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.detrend, input, axis=2)

    def test_demean_2D_d2_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.demean, input, axis=2)

    def test_detrend_linear_0D_zeros(self):
        input = 0.
        targ = 0.
        res = mlab.detrend_linear(input)
        assert_almost_equal(res, targ)

    def test_detrend_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend_linear(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend(input, key='linear')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = mlab.detrend(input, key=mlab.detrend_linear)
        assert_almost_equal(res, targ)

    def test_detrend_linear_1d_off(self):
        input = self.sig_off
        targ = self.sig_zeros
        res = mlab.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope(self):
        input = self.sig_slope
        targ = self.sig_zeros
        res = mlab.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = mlab.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = mlab.detrend(input, key='linear')
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = mlab.detrend(input, key=mlab.detrend_linear)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off_list(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = mlab.detrend_linear(input.tolist())
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_2D_ValueError(self):
        input = self.sig_slope[np.newaxis]
        assert_raises(ValueError, mlab.detrend_linear, input)

    def test_detrend_str_linear_2d_slope_off_axis0(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend(input, key='linear', axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = mlab.detrend(input, key=mlab.detrend_linear, axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_2d_slope_off_axis0(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend(input, key='linear', axis=1)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = mlab.detrend(input, key=mlab.detrend_linear, axis=1)
        assert_allclose(res, targ, atol=self.atol)


class spectral_testcase_nosig_real_onesided(CleanupTestCase):
    def setUp(self):
        self.createStim(fstims=[],
                        iscomplex=False, sides='onesided', nsides=1)

    def createStim(self, fstims, iscomplex, sides, nsides, len_x=None,
                   NFFT_density=-1, nover_density=-1, pad_to_density=-1,
                   pad_to_spectrum=-1):
        Fs = 100.

        x = np.arange(0, 10, 1/Fs)
        if len_x is not None:
            x = x[:len_x]

        # get the stimulus frequencies, defaulting to None
        fstims = [Fs/fstim for fstim in fstims]

        # get the constants, default to calculated values
        if NFFT_density is None:
            NFFT_density_real = 256
        elif NFFT_density < 0:
            NFFT_density_real = NFFT_density = 100
        else:
            NFFT_density_real = NFFT_density

        if nover_density is None:
            nover_density_real = 0
        elif nover_density < 0:
            nover_density_real = nover_density = NFFT_density_real//2
        else:
            nover_density_real = nover_density

        if pad_to_density is None:
            pad_to_density_real = NFFT_density_real
        elif pad_to_density < 0:
            pad_to_density = int(2**np.ceil(np.log2(NFFT_density_real)))
            pad_to_density_real = pad_to_density
        else:
            pad_to_density_real = pad_to_density

        if pad_to_spectrum is None:
            pad_to_spectrum_real = len(x)
        elif pad_to_spectrum < 0:
            pad_to_spectrum_real = pad_to_spectrum = len(x)
        else:
            pad_to_spectrum_real = pad_to_spectrum

        if pad_to_spectrum is None:
            NFFT_spectrum_real = NFFT_spectrum = pad_to_spectrum_real
        else:
            NFFT_spectrum_real = NFFT_spectrum = len(x)
        nover_spectrum_real = nover_spectrum = 0

        NFFT_specgram = NFFT_density
        nover_specgram = nover_density
        pad_to_specgram = pad_to_density
        NFFT_specgram_real = NFFT_density_real
        nover_specgram_real = nover_density_real

        if nsides == 1:
            # frequencies for specgram, psd, and csd
            # need to handle even and odd differently
            if pad_to_density_real % 2:
                freqs_density = np.linspace(0, Fs/2,
                                            num=pad_to_density_real,
                                            endpoint=False)[::2]
            else:
                freqs_density = np.linspace(0, Fs/2,
                                            num=pad_to_density_real//2+1)

            # frequencies for complex, magnitude, angle, and phase spectrums
            # need to handle even and odd differently
            if pad_to_spectrum_real % 2:
                freqs_spectrum = np.linspace(0, Fs/2,
                                             num=pad_to_spectrum_real,
                                             endpoint=False)[::2]
            else:
                freqs_spectrum = np.linspace(0, Fs/2,
                                             num=pad_to_spectrum_real//2+1)
        else:
            # frequencies for specgram, psd, and csd
            # need to handle even and odd differentl
            if pad_to_density_real % 2:
                freqs_density = np.linspace(-Fs/2, Fs/2,
                                            num=2*pad_to_density_real,
                                            endpoint=False)[1::2]
            else:
                freqs_density = np.linspace(-Fs/2, Fs/2,
                                            num=pad_to_density_real,
                                            endpoint=False)

            # frequencies for complex, magnitude, angle, and phase spectrums
            # need to handle even and odd differently
            if pad_to_spectrum_real % 2:
                freqs_spectrum = np.linspace(-Fs/2, Fs/2,
                                             num=2*pad_to_spectrum_real,
                                             endpoint=False)[1::2]
            else:
                freqs_spectrum = np.linspace(-Fs/2, Fs/2,
                                             num=pad_to_spectrum_real,
                                             endpoint=False)

        freqs_specgram = freqs_density
        # time points for specgram
        t_start = NFFT_specgram_real//2
        t_stop = len(x) - NFFT_specgram_real//2+1
        t_step = NFFT_specgram_real - nover_specgram_real
        t_specgram = x[t_start:t_stop:t_step]
        if NFFT_specgram_real % 2:
            t_specgram += 1/Fs/2
        if len(t_specgram) == 0:
            t_specgram = np.array([NFFT_specgram_real/(2*Fs)])
        t_spectrum = np.array([NFFT_spectrum_real/(2*Fs)])
        t_density = t_specgram

        y = np.zeros_like(x)
        for i, fstim in enumerate(fstims):
            y += np.sin(fstim * x * np.pi * 2) * 10**i

        if iscomplex:
            y = y.astype('complex')

        self.Fs = Fs
        self.sides = sides
        self.fstims = fstims

        self.NFFT_density = NFFT_density
        self.nover_density = nover_density
        self.pad_to_density = pad_to_density

        self.NFFT_spectrum = NFFT_spectrum
        self.nover_spectrum = nover_spectrum
        self.pad_to_spectrum = pad_to_spectrum

        self.NFFT_specgram = NFFT_specgram
        self.nover_specgram = nover_specgram
        self.pad_to_specgram = pad_to_specgram

        self.t_specgram = t_specgram
        self.t_density = t_density
        self.t_spectrum = t_spectrum
        self.y = y

        self.freqs_density = freqs_density
        self.freqs_spectrum = freqs_spectrum
        self.freqs_specgram = freqs_specgram

        self.NFFT_density_real = NFFT_density_real

    def check_freqs(self, vals, targfreqs, resfreqs, fstims):
        assert_true(resfreqs.argmin() == 0)
        assert_true(resfreqs.argmax() == len(resfreqs)-1)
        assert_allclose(resfreqs, targfreqs, atol=1e-06)
        for fstim in fstims:
            i = np.abs(resfreqs - fstim).argmin()
            assert_true(vals[i] > vals[i+2])
            assert_true(vals[i] > vals[i-2])

    def check_maxfreq(self, spec, fsp, fstims):
        # skip the test if there are no frequencies
        if len(fstims) == 0:
            return

        # if twosided, do the test for each side
        if fsp.min() < 0:
            fspa = np.abs(fsp)
            zeroind = fspa.argmin()
            self.check_maxfreq(spec[:zeroind], fspa[:zeroind], fstims)
            self.check_maxfreq(spec[zeroind:], fspa[zeroind:], fstims)
            return

        fstimst = fstims[:]
        spect = spec.copy()

        # go through each peak and make sure it is correctly the maximum peak
        while fstimst:
            maxind = spect.argmax()
            maxfreq = fsp[maxind]
            assert_almost_equal(maxfreq, fstimst[-1])
            del fstimst[-1]
            spect[maxind-5:maxind+5] = 0

    def test_spectral_helper_raises_complex_same_data(self):
        # test that mode 'complex' cannot be used if x is not y
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y+1, mode='complex')

    def test_spectral_helper_raises_magnitude_same_data(self):
        # test that mode 'magnitude' cannot be used if x is not y
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y+1, mode='magnitude')

    def test_spectral_helper_raises_angle_same_data(self):
        # test that mode 'angle' cannot be used if x is not y
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y+1, mode='angle')

    def test_spectral_helper_raises_phase_same_data(self):
        # test that mode 'phase' cannot be used if x is not y
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y+1, mode='phase')

    def test_spectral_helper_raises_unknown_mode(self):
        # test that unknown value for mode cannot be used
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, mode='spam')

    def test_spectral_helper_raises_unknown_sides(self):
        # test that unknown value for sides cannot be used
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y, sides='eggs')

    def test_spectral_helper_raises_noverlap_gt_NFFT(self):
        # test that noverlap cannot be larger than NFFT
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y, NFFT=10, noverlap=20)

    def test_spectral_helper_raises_noverlap_eq_NFFT(self):
        # test that noverlap cannot be equal to NFFT
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, NFFT=10, noverlap=10)

    def test_spectral_helper_raises_winlen_ne_NFFT(self):
        # test that the window length cannot be different from NFFT
        assert_raises(ValueError, mlab._spectral_helper,
                      x=self.y, y=self.y, NFFT=10, window=np.ones(9))

    def test_single_spectrum_helper_raises_mode_default(self):
        # test that mode 'default' cannot be used with _single_spectrum_helper
        assert_raises(ValueError, mlab._single_spectrum_helper,
                      x=self.y, mode='default')

    def test_single_spectrum_helper_raises_mode_psd(self):
        # test that mode 'psd' cannot be used with _single_spectrum_helper
        assert_raises(ValueError, mlab._single_spectrum_helper,
                      x=self.y, mode='psd')

    def test_spectral_helper_psd(self):
        freqs = self.freqs_density
        spec, fsp, t = mlab._spectral_helper(x=self.y, y=self.y,
                                             NFFT=self.NFFT_density,
                                             Fs=self.Fs,
                                             noverlap=self.nover_density,
                                             pad_to=self.pad_to_density,
                                             sides=self.sides,
                                             mode='psd')

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_density, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

    def test_spectral_helper_magnitude_specgram(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab._spectral_helper(x=self.y, y=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='magnitude')

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

    def test_spectral_helper_magnitude_magnitude_spectrum(self):
        freqs = self.freqs_spectrum
        spec, fsp, t = mlab._spectral_helper(x=self.y, y=self.y,
                                             NFFT=self.NFFT_spectrum,
                                             Fs=self.Fs,
                                             noverlap=self.nover_spectrum,
                                             pad_to=self.pad_to_spectrum,
                                             sides=self.sides,
                                             mode='magnitude')

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_spectrum, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], 1)

    def test_csd(self):
        freqs = self.freqs_density
        spec, fsp = mlab.csd(x=self.y, y=self.y+1,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_equal(spec.shape, freqs.shape)

    def test_psd(self):
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides)
        assert_equal(spec.shape, freqs.shape)
        self.check_freqs(spec, freqs, fsp, self.fstims)

    def test_psd_detrend_mean_func_offset(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.zeros(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ydata = np.vstack([ydata1, ydata2])
        ydata = np.tile(ydata, (20, 1))
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        ycontrol = np.zeros_like(ydata)
        spec_g, fsp_g = mlab.psd(x=ydata,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_mean)
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_mean)
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides)
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_detrend_mean_str_offset(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.zeros(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ydata = np.vstack([ydata1, ydata2])
        ydata = np.tile(ydata, (20, 1))
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        ycontrol = np.zeros_like(ydata)
        spec_g, fsp_g = mlab.psd(x=ydata,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend='mean')
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend='mean')
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides)
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_detrend_linear_func_trend(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.arange(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ydata = np.vstack([ydata1, ydata2])
        ydata = np.tile(ydata, (20, 1))
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        ycontrol = np.zeros_like(ydata)
        spec_g, fsp_g = mlab.psd(x=ydata,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear)
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear)
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides)
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_detrend_linear_str_trend(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.arange(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ydata = np.vstack([ydata1, ydata2])
        ydata = np.tile(ydata, (20, 1))
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        ycontrol = np.zeros_like(ydata)
        spec_g, fsp_g = mlab.psd(x=ydata,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend='linear')
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend='linear')
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides)
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_window_hanning(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.arange(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ycontrol1, windowVals = mlab.apply_window(ydata1,
                                                  mlab.window_hanning,
                                                  return_window=True)
        ycontrol2 = mlab.window_hanning(ydata2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        spec_g, fsp_g = mlab.psd(x=ydataf,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_hanning)
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_hanning)
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_none)
        spec_c *= len(ycontrol1)/(np.abs(windowVals)**2).sum()
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_window_hanning_detrend_linear(self):
        if self.NFFT_density is None:
            return
        freqs = self.freqs_density
        ydata = np.arange(self.NFFT_density)
        ycontrol = np.zeros(self.NFFT_density)
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        ycontrol1 = ycontrol
        ycontrol2 = ycontrol
        ycontrol1, windowVals = mlab.apply_window(ycontrol1,
                                                  mlab.window_hanning,
                                                  return_window=True)
        ycontrol2 = mlab.window_hanning(ycontrol2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        spec_g, fsp_g = mlab.psd(x=ydataf,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear,
                                 window=mlab.window_hanning)
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear,
                                 window=mlab.window_hanning)
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_none)
        spec_c *= len(ycontrol1)/(np.abs(windowVals)**2).sum()
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # these should not be almost equal
        assert_raises(AssertionError,
                      assert_allclose, spec_b, spec_c, atol=1e-08)

    def test_psd_windowarray(self):
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides,
                             window=np.ones(self.NFFT_density_real))
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_equal(spec.shape, freqs.shape)

    def test_psd_windowarray_scale_by_freq(self):
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides)
        spec_s, fsp_s = mlab.psd(x=self.y,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=self.nover_density,
                                 pad_to=self.pad_to_density,
                                 sides=self.sides,
                                 scale_by_freq=True)
        spec_n, fsp_n = mlab.psd(x=self.y,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=self.nover_density,
                                 pad_to=self.pad_to_density,
                                 sides=self.sides,
                                 scale_by_freq=False)

        assert_array_equal(fsp, fsp_s)
        assert_array_equal(fsp, fsp_n)
        assert_array_equal(spec, spec_s)
        assert_allclose(spec_s, spec_n/self.Fs, atol=1e-08)

    def test_complex_spectrum(self):
        freqs = self.freqs_spectrum
        spec, fsp = mlab.complex_spectrum(x=self.y,
                                          Fs=self.Fs,
                                          sides=self.sides,
                                          pad_to=self.pad_to_spectrum)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_equal(spec.shape, freqs.shape)

    def test_magnitude_spectrum(self):
        freqs = self.freqs_spectrum
        spec, fsp = mlab.magnitude_spectrum(x=self.y,
                                            Fs=self.Fs,
                                            sides=self.sides,
                                            pad_to=self.pad_to_spectrum)
        assert_equal(spec.shape, freqs.shape)
        self.check_maxfreq(spec, fsp, self.fstims)
        self.check_freqs(spec, freqs, fsp, self.fstims)

    def test_angle_spectrum(self):
        freqs = self.freqs_spectrum
        spec, fsp = mlab.angle_spectrum(x=self.y,
                                        Fs=self.Fs,
                                        sides=self.sides,
                                        pad_to=self.pad_to_spectrum)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_equal(spec.shape, freqs.shape)

    def test_phase_spectrum(self):
        freqs = self.freqs_spectrum
        spec, fsp = mlab.phase_spectrum(x=self.y,
                                        Fs=self.Fs,
                                        sides=self.sides,
                                        pad_to=self.pad_to_spectrum)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_equal(spec.shape, freqs.shape)

    def test_specgram_auto(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides)
        specm = np.mean(spec, axis=1)

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

        # since we are using a single freq, all time slices
        # should be about the same
        if np.abs(spec.max()) != 0:
            assert_allclose(np.diff(spec, axis=1).max()/np.abs(spec.max()), 0,
                            atol=1e-02)
        self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_default(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='default')
        specm = np.mean(spec, axis=1)

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

        # since we are using a single freq, all time slices
        # should be about the same
        if np.abs(spec.max()) != 0:
            assert_allclose(np.diff(spec, axis=1).max()/np.abs(spec.max()), 0,
                            atol=1e-02)
        self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_psd(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='psd')
        specm = np.mean(spec, axis=1)

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])
        # since we are using a single freq, all time slices
        # should be about the same
        if np.abs(spec.max()) != 0:
            assert_allclose(np.diff(spec, axis=1).max()/np.abs(spec.max()), 0,
                            atol=1e-02)
        self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_complex(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='complex')
        specm = np.mean(np.abs(spec), axis=1)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

        self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_magnitude(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='magnitude')
        specm = np.mean(spec, axis=1)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])
        # since we are using a single freq, all time slices
        # should be about the same
        if np.abs(spec.max()) != 0:
            assert_allclose(np.diff(spec, axis=1).max()/np.abs(spec.max()), 0,
                            atol=1e-02)
        self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_angle(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='angle')
        specm = np.mean(spec, axis=1)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

    def test_specgram_phase(self):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     mode='phase')
        specm = np.mean(spec, axis=1)

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)

        assert_equal(spec.shape[0], freqs.shape[0])
        assert_equal(spec.shape[1], self.t_specgram.shape[0])

    def test_psd_csd_equal(self):
        freqs = self.freqs_density
        Pxx, freqsxx = mlab.psd(x=self.y,
                                NFFT=self.NFFT_density,
                                Fs=self.Fs,
                                noverlap=self.nover_density,
                                pad_to=self.pad_to_density,
                                sides=self.sides)
        Pxy, freqsxy = mlab.csd(x=self.y, y=self.y,
                                NFFT=self.NFFT_density,
                                Fs=self.Fs,
                                noverlap=self.nover_density,
                                pad_to=self.pad_to_density,
                                sides=self.sides)
        assert_array_equal(Pxx, Pxy)
        assert_array_equal(freqsxx, freqsxy)

    def test_specgram_auto_default_equal(self):
        '''test that mlab.specgram without mode and with mode 'default' and
        'psd' are all the same'''
        freqs = self.freqs_specgram
        speca, freqspeca, ta = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides)
        specb, freqspecb, tb = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='default')
        assert_array_equal(speca, specb)
        assert_array_equal(freqspeca, freqspecb)
        assert_array_equal(ta, tb)

    def test_specgram_auto_psd_equal(self):
        '''test that mlab.specgram without mode and with mode 'default' and
        'psd' are all the same'''
        freqs = self.freqs_specgram
        speca, freqspeca, ta = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides)
        specc, freqspecc, tc = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='psd')
        assert_array_equal(speca, specc)
        assert_array_equal(freqspeca, freqspecc)
        assert_array_equal(ta, tc)

    def test_specgram_complex_mag_equivalent(self):
        freqs = self.freqs_specgram
        specc, freqspecc, tc = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='complex')
        specm, freqspecm, tm = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='magnitude')

        assert_array_equal(freqspecc, freqspecm)
        assert_array_equal(tc, tm)
        assert_allclose(np.abs(specc), specm, atol=1e-06)

    def test_specgram_complex_angle_equivalent(self):
        freqs = self.freqs_specgram
        specc, freqspecc, tc = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='complex')
        speca, freqspeca, ta = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='angle')

        assert_array_equal(freqspecc, freqspeca)
        assert_array_equal(tc, ta)
        assert_allclose(np.angle(specc), speca, atol=1e-06)

    def test_specgram_complex_phase_equivalent(self):
        freqs = self.freqs_specgram
        specc, freqspecc, tc = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='complex')
        specp, freqspecp, tp = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='phase')

        assert_array_equal(freqspecc, freqspecp)
        assert_array_equal(tc, tp)
        assert_allclose(np.unwrap(np.angle(specc), axis=0), specp,
                        atol=1e-06)

    def test_specgram_angle_phase_equivalent(self):
        freqs = self.freqs_specgram
        speca, freqspeca, ta = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='angle')
        specp, freqspecp, tp = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='phase')

        assert_array_equal(freqspeca, freqspecp)
        assert_array_equal(ta, tp)
        assert_allclose(np.unwrap(speca, axis=0), specp,
                        atol=1e-06)

    def test_psd_windowarray_equal(self):
        freqs = self.freqs_density
        win = mlab.window_hanning(np.ones(self.NFFT_density_real))
        speca, fspa = mlab.psd(x=self.y,
                               NFFT=self.NFFT_density,
                               Fs=self.Fs,
                               noverlap=self.nover_density,
                               pad_to=self.pad_to_density,
                               sides=self.sides,
                               window=win)
        specb, fspb = mlab.psd(x=self.y,
                               NFFT=self.NFFT_density,
                               Fs=self.Fs,
                               noverlap=self.nover_density,
                               pad_to=self.pad_to_density,
                               sides=self.sides)
        assert_array_equal(fspa, fspb)
        assert_allclose(speca, specb, atol=1e-08)


class spectral_testcase_nosig_real_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_Fs4_real_onesided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_Fs4_real_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_Fs4_real_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_Fs4_complex_onesided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_Fs4_complex_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_Fs4_complex_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4],
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_FsAll_real_onesided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_FsAll_real_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_FsAll_real_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_FsAll_complex_onesided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_FsAll_complex_twosided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_FsAll_complex_defaultsided(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[4, 5, 10],
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_noNFFT(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_nopad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_noNFFT_no_pad_to(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                NFFT_density=None,
                                pad_to_density=None, pad_to_spectrum=None,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=512, pad_to_spectrum=128,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=512, pad_to_spectrum=128,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=512, pad_to_spectrum=128,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=512, pad_to_spectrum=128,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=512, pad_to_spectrum=128,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_trim(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                NFFT_density=128, pad_to_spectrum=128,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_odd(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=256,
                                pad_to_density=33, pad_to_spectrum=257,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=33, pad_to_spectrum=None,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=33, pad_to_spectrum=None,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=33, pad_to_spectrum=None,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=33, pad_to_spectrum=None,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=33, pad_to_spectrum=None,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_oddlen(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=255,
                                NFFT_density=128, pad_to_spectrum=None,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_stretch(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                len_x=128,
                                NFFT_density=128,
                                pad_to_density=256, pad_to_spectrum=256,
                                iscomplex=True, sides='default', nsides=2)


class spectral_testcase_nosig_real_onesided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=False, sides='onesided', nsides=1)


class spectral_testcase_nosig_real_twosided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=False, sides='twosided', nsides=2)


class spectral_testcase_nosig_real_defaultsided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=False, sides='default', nsides=1)


class spectral_testcase_nosig_complex_onesided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=True, sides='onesided', nsides=1)


class spectral_testcase_nosig_complex_twosided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=True, sides='twosided', nsides=2)


class spectral_testcase_nosig_complex_defaultsided_overlap(
        spectral_testcase_nosig_real_onesided):
        def setUp(self):
                self.createStim(fstims=[],
                                nover_density=32,
                                iscomplex=True, sides='default', nsides=2)


def test_griddata_linear():
    # z is a linear function of x and y.
    def get_z(x, y):
        return 3.0*x - y

    # Passing 1D xi and yi arrays to griddata.
    x = np.asarray([0.0, 1.0, 0.0, 1.0, 0.5])
    y = np.asarray([0.0, 0.0, 1.0, 1.0, 0.5])
    z = get_z(x, y)
    xi = [0.2, 0.4, 0.6, 0.8]
    yi = [0.1, 0.3, 0.7, 0.9]
    zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
    xi, yi = np.meshgrid(xi, yi)
    np.testing.assert_array_almost_equal(zi, get_z(xi, yi))

    # Passing 2D xi and yi arrays to griddata.
    zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
    np.testing.assert_array_almost_equal(zi, get_z(xi, yi))

    # Masking z array.
    z_masked = np.ma.array(z, mask=[False, False, False, True, False])
    correct_zi_masked = np.ma.masked_where(xi + yi > 1.0, get_z(xi, yi))
    zi = mlab.griddata(x, y, z_masked, xi, yi, interp='linear')
    matest.assert_array_almost_equal(zi, correct_zi_masked)
    np.testing.assert_array_equal(np.ma.getmask(zi),
                                  np.ma.getmask(correct_zi_masked))


@knownfailureif(not HAS_NATGRID)
def test_griddata_nn():
    # z is a linear function of x and y.
    def get_z(x, y):
        return 3.0*x - y

    # Passing 1D xi and yi arrays to griddata.
    x = np.asarray([0.0, 1.0, 0.0, 1.0, 0.5])
    y = np.asarray([0.0, 0.0, 1.0, 1.0, 0.5])
    z = get_z(x, y)
    xi = [0.2, 0.4, 0.6, 0.8]
    yi = [0.1, 0.3, 0.7, 0.9]
    correct_zi = [[0.49999252, 1.0999978, 1.7000030, 2.3000080],
                  [0.29999208, 0.8999978, 1.5000029, 2.1000059],
                  [-0.1000099, 0.4999943, 1.0999964, 1.6999979],
                  [-0.3000128, 0.2999894, 0.8999913, 1.4999933]]
    zi = mlab.griddata(x, y, z, xi, yi, interp='nn')
    np.testing.assert_array_almost_equal(zi, correct_zi)

    # Decreasing xi or yi should raise ValueError.
    assert_raises(ValueError, mlab.griddata, x, y, z, xi[::-1], yi,
                  interp='nn')
    assert_raises(ValueError, mlab.griddata, x, y, z, xi, yi[::-1],
                  interp='nn')

    # Passing 2D xi and yi arrays to griddata.
    xi, yi = np.meshgrid(xi, yi)
    zi = mlab.griddata(x, y, z, xi, yi, interp='nn')
    np.testing.assert_array_almost_equal(zi, correct_zi)

    # Masking z array.
    z_masked = np.ma.array(z, mask=[False, False, False, True, False])
    correct_zi_masked = np.ma.masked_where(xi + yi > 1.0, correct_zi)
    zi = mlab.griddata(x, y, z_masked, xi, yi, interp='nn')
    np.testing.assert_array_almost_equal(zi, correct_zi_masked, 5)
    np.testing.assert_array_equal(np.ma.getmask(zi),
                                  np.ma.getmask(correct_zi_masked))


#*****************************************************************
# These Tests where taken from SCIPY with some minor modifications
# this can be retreived from:
# https://github.com/scipy/scipy/blob/master/scipy/stats/tests/test_kdeoth.py
#*****************************************************************

class gaussian_kde_tests():

    def test_kde_integer_input(self):
        """Regression test for #1181."""
        x1 = np.arange(5)
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869,
                      0.13480721]
        np.testing.assert_array_almost_equal(kde(x1), y_expected, decimal=6)

    def test_gaussian_kde_covariance_caching(self):
        x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float)
        xs = np.linspace(-10, 10, num=5)
        # These expected values are from scipy 0.10, before some changes to
        # gaussian_kde. They were not compared with any external reference.
        y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754,
                      0.01664475]

        # set it to the default bandwidth.
        kde2 = mlab.GaussianKDE(x1, 'scott')
        y2 = kde2(xs)

        np.testing.assert_array_almost_equal(y_expected, y2, decimal=7)

    def test_kde_bandwidth_method(self):

        np.random.seed(8765678)
        n_basesample = 50
        xn = np.random.randn(n_basesample)

        # Default
        gkde = mlab.GaussianKDE(xn)
        # Supply a callable
        gkde2 = mlab.GaussianKDE(xn, 'scott')
        # Supply a scalar
        gkde3 = mlab.GaussianKDE(xn, bw_method=gkde.factor)

        xs = np.linspace(-7, 7, 51)
        kdepdf = gkde.evaluate(xs)
        kdepdf2 = gkde2.evaluate(xs)
        assert_almost_equal(kdepdf.all(), kdepdf2.all())
        kdepdf3 = gkde3.evaluate(xs)
        assert_almost_equal(kdepdf.all(), kdepdf3.all())


class gaussian_kde_custom_tests(object):
    def test_no_data(self):
        """Pass no data into the GaussianKDE class."""
        assert_raises(ValueError, mlab.GaussianKDE, [])

    def test_single_dataset_element(self):
        """Pass a single dataset element into the GaussianKDE class."""
        assert_raises(ValueError, mlab.GaussianKDE, [42])

    def test_silverman_multidim_dataset(self):
        """Use a multi-dimensional array as the dataset and test silverman's
        output"""
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_raises(np.linalg.LinAlgError, mlab.GaussianKDE, x1, "silverman")

    def test_silverman_singledim_dataset(self):
        """Use a single dimension list as the dataset and test silverman's
        output."""
        x1 = np.array([-7, -5, 1, 4, 5])
        mygauss = mlab.GaussianKDE(x1, "silverman")
        y_expected = 0.76770389927475502
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    def test_scott_multidim_dataset(self):
        """Use a multi-dimensional array as the dataset and test scott's output
        """
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_raises(np.linalg.LinAlgError, mlab.GaussianKDE, x1, "scott")

    def test_scott_singledim_dataset(self):
        """Use a single-dimensional array as the dataset and test scott's
        output"""
        x1 = np.array([-7, -5, 1, 4, 5])
        mygauss = mlab.GaussianKDE(x1, "scott")
        y_expected = 0.72477966367769553
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    def test_scalar_empty_dataset(self):
        """Use an empty array as the dataset and test the scalar's cov factor
        """
        assert_raises(ValueError, mlab.GaussianKDE, [], bw_method=5)

    def test_scalar_covariance_dataset(self):
        """Use a dataset and test a scalar's cov factor
        """
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]

        kde = mlab.GaussianKDE(multidim_data, bw_method=0.5)
        assert_equal(kde.covariance_factor(), 0.5)

    def test_callable_covariance_dataset(self):
        """Use a multi-dimensional array as the dataset and test the callable's
        cov factor"""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]
        callable_fun = lambda x: 0.55
        kde = mlab.GaussianKDE(multidim_data, bw_method=callable_fun)
        assert_equal(kde.covariance_factor(), 0.55)

    def test_callable_singledim_dataset(self):
        """Use a single-dimensional array as the dataset and test the
        callable's cov factor"""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = np.random.randn(n_basesample)

        kde = mlab.GaussianKDE(multidim_data, bw_method='silverman')
        y_expected = 0.48438841363348911
        assert_almost_equal(kde.covariance_factor(), y_expected, 7)

    def test_wrong_bw_method(self):
        """Test the error message that should be called when bw is invalid."""
        np.random.seed(8765678)
        n_basesample = 50
        data = np.random.randn(n_basesample)
        assert_raises(ValueError, mlab.GaussianKDE, data, bw_method="invalid")


class gaussian_kde_evaluate_tests(object):

    def test_evaluate_diff_dim(self):
        """Test the evaluate method when the dim's of dataset and points are
        different dimensions"""
        x1 = np.arange(3, 10, 2)
        kde = mlab.GaussianKDE(x1)
        x2 = np.arange(3, 12, 2)
        y_expected = [
            0.08797252, 0.11774109, 0.11774109, 0.08797252, 0.0370153
        ]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_inv_dim(self):
        """ Invert the dimensions. i.e., Give the dataset a dimension of
        1 [3,2,4], and the points will have a dimension of 3 [[3],[2],[4]].
        ValueError should be raised"""
        np.random.seed(8765678)
        n_basesample = 50
        multidim_data = np.random.randn(n_basesample)
        kde = mlab.GaussianKDE(multidim_data)
        x2 = [[1], [2], [3]]
        assert_raises(ValueError, kde.evaluate, x2)

    def test_evaluate_dim_and_num(self):
        """ Tests if evaluated against a one by one array"""
        x1 = np.arange(3, 10, 2)
        x2 = np.array([3])
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.08797252]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_point_dim_not_one(self):
        """Test"""
        x1 = np.arange(3, 10, 2)
        x2 = [np.arange(3, 10, 2), np.arange(3, 10, 2)]
        kde = mlab.GaussianKDE(x1)
        assert_raises(ValueError, kde.evaluate, x2)

    def test_evaluate_equal_dim_and_num_lt(self):
        """Test when line 3810 fails"""
        x1 = np.arange(3, 10, 2)
        x2 = np.arange(3, 8, 2)
        kde = mlab.GaussianKDE(x1)
        y_expected = [0.08797252, 0.11774109, 0.11774109]
        y = kde.evaluate(x2)
        np.testing.assert_array_almost_equal(y, y_expected, 7)


#*****************************************************************
#*****************************************************************

if __name__ == '__main__':
    import nose
    import sys

    args = ['-s', '--with-doctest']
    argv = sys.argv
    argv = argv[:1] + args + argv[1:]
    nose.runmodule(argv=argv, exit=False)
