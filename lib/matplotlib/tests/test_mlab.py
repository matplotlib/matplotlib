from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import sys

import numpy as np
import matplotlib.mlab as mlab
import tempfile
import unittest

from numpy.testing import assert_allclose, assert_array_equal


class general_testcase(unittest.TestCase):
    def test_colinear_pca(self):
        a = mlab.PCA._get_colinear()
        pca = mlab.PCA(a)

        assert_allclose(pca.fracs[2:], 0., atol=1e-8)
        assert_allclose(pca.Y[:, 2:], 0., atol=1e-8)

    def test_prctile(self):
        # test odd lengths
        x = [1, 2, 3]
        self.assertEqual(mlab.prctile(x, 50), np.median(x))

        # test even lengths
        x = [1, 2, 3, 4]
        self.assertEqual(mlab.prctile(x, 50), np.median(x))

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


class csv_testcase(unittest.TestCase):
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
        self.assertRaises(ValueError, mlab.rec2csv, bad, self.fd)


class window_testcase(unittest.TestCase):
    '''Check window-related functions'''
    def setUp(self):
        '''shared set-up code for window tests'''
        np.random.seed(0)
        self.n = 100000
        self.x = np.arange(0., self.n)

        self.sig_rand = np.random.standard_normal(self.n) + 100.
        self.sig_ones = np.ones_like(self.x)
        self.sig_slope = np.linspace(-10., 90., self.n)

    def test_window_none(self):
        '''test mlab.window_none'''
        res_rand = mlab.window_none(self.sig_rand)
        res_ones = mlab.window_none(self.sig_ones)

        assert_array_equal(res_rand, self.sig_rand)
        assert_array_equal(res_ones, self.sig_ones)

    def test_window_hanning(self):
        '''test mlab.window_hanning'''
        targ_rand = np.hanning(len(self.sig_rand)) * self.sig_rand
        res_rand = mlab.window_hanning(self.sig_rand)

        targ_ones = np.hanning(len(self.sig_ones))
        res_ones = mlab.window_hanning(self.sig_ones)

        assert_allclose(targ_rand, res_rand, atol=1e-06)
        assert_allclose(targ_ones, res_ones, atol=1e-06)


class detrend_testcase(unittest.TestCase):
    '''Check detrend-related functions'''
    def setUp(self):
        '''shared set-up code for detrend tests'''
        np.random.seed(0)
        n = 100000
        x = np.arange(0., n)

        self.sig_zeros = np.zeros(n)

        self.sig_off = self.sig_zeros + 100.
        self.sig_slope = np.linspace(-10., 90., n)

        self.sig_slope_mean = np.linspace(-50, 50, n)

        self.sig_rand = np.random.standard_normal(n)
        self.sig_sin = np.sin(x*2*np.pi/(n/100))

        self.sig_rand -= self.sig_rand.mean()
        self.sig_sin -= self.sig_sin.mean()

        self.sig_slope_off = self.sig_slope + self.sig_off
        self.sig_sin_rand = self.sig_sin + self.sig_rand

        self.sig_rand_off = self.sig_rand + self.sig_off
        self.sig_sin_off = self.sig_sin + self.sig_off
        self.sig_sin_rand_off = self.sig_sin_rand + self.sig_off

        self.sig_rand_slope = self.sig_rand + self.sig_slope
        self.sig_sin_slope = self.sig_sin + self.sig_slope
        self.sig_sin_rand_slope = self.sig_sin_rand + self.sig_slope

        self.sig_rand_slope_mean = self.sig_rand + self.sig_slope_mean
        self.sig_sin_slope_mean = self.sig_sin + self.sig_slope_mean
        self.sig_sin_rand_slope_mean = self.sig_sin_rand + self.sig_slope_mean

        self.sig_rand_slope_off = self.sig_rand + self.sig_slope_off
        self.sig_sin_slope_off = self.sig_sin + self.sig_slope_off
        self.sig_sin_rand_slope_off = self.sig_sin_rand + self.sig_slope_off

    def test_detrend_none(self):
        '''test mlab.detrend_none'''
        res_off = mlab.detrend_none(self.sig_off)
        res_slope = mlab.detrend_none(self.sig_slope)

        res_rand = mlab.detrend_none(self.sig_rand)
        res_sin = mlab.detrend_none(self.sig_sin)

        res_slope_off = mlab.detrend_none(self.sig_slope_off)
        res_sin_rand = mlab.detrend_none(self.sig_sin_rand)

        res_rand_off = mlab.detrend_none(self.sig_rand_off)
        res_sin_off = mlab.detrend_none(self.sig_sin_off)
        res_sin_rand_off = mlab.detrend_none(self.sig_sin_rand_off)

        res_rand_slope = mlab.detrend_none(self.sig_rand_slope)
        res_sin_slope = mlab.detrend_none(self.sig_sin_slope)
        res_sin_rand_slope = mlab.detrend_none(self.sig_sin_rand_slope)

        res_rand_slope_off = mlab.detrend_none(self.sig_rand_slope_off)
        res_sin_slope_off = mlab.detrend_none(self.sig_sin_slope_off)
        res_sin_rand_slope_off = mlab.detrend_none(self.sig_sin_rand_slope_off)

        assert_array_equal(res_off, self.sig_off)
        assert_array_equal(res_slope, self.sig_slope)

        assert_array_equal(res_rand, self.sig_rand)
        assert_array_equal(res_sin, self.sig_sin)

        assert_array_equal(res_slope_off, self.sig_slope_off)
        assert_array_equal(res_sin_rand, self.sig_sin_rand)

        assert_array_equal(res_rand_off, self.sig_rand_off)
        assert_array_equal(res_sin_off, self.sig_sin_off)
        assert_array_equal(res_sin_rand_off, self.sig_sin_rand_off)

        assert_array_equal(res_rand_slope, self.sig_rand_slope)
        assert_array_equal(res_sin_slope, self.sig_sin_slope)
        assert_array_equal(res_sin_rand_slope, self.sig_sin_rand_slope)

        assert_array_equal(res_rand_slope_off, self.sig_rand_slope_off)
        assert_array_equal(res_sin_slope_off, self.sig_sin_slope_off)
        assert_array_equal(res_sin_rand_slope_off, self.sig_sin_rand_slope_off)

    def test_detrend_mean(self):
        '''test mlab.detrend_none'''
        res_off = mlab.detrend_mean(self.sig_off)
        res_slope = mlab.detrend_mean(self.sig_slope)

        res_rand = mlab.detrend_mean(self.sig_rand)
        res_sin = mlab.detrend_mean(self.sig_sin)

        res_slope_off = mlab.detrend_mean(self.sig_slope_off)
        res_sin_rand = mlab.detrend_mean(self.sig_sin_rand)

        res_rand_off = mlab.detrend_mean(self.sig_rand_off)
        res_sin_off = mlab.detrend_mean(self.sig_sin_off)
        res_sin_rand_off = mlab.detrend_mean(self.sig_sin_rand_off)

        res_rand_slope = mlab.detrend_mean(self.sig_rand_slope)
        res_sin_slope = mlab.detrend_mean(self.sig_sin_slope)
        res_sin_rand_slope = mlab.detrend_mean(self.sig_sin_rand_slope)

        res_rand_slope_off = mlab.detrend_mean(self.sig_rand_slope_off)
        res_sin_slope_off = mlab.detrend_mean(self.sig_sin_slope_off)
        res_sin_rand_slope_off = mlab.detrend_mean(self.sig_sin_rand_slope_off)

        assert_allclose(res_off, self.sig_zeros, atol=1e-06)
        assert_allclose(res_slope, self.sig_slope_mean, atol=1e-06)

        assert_allclose(res_rand, self.sig_rand, atol=1e-06)
        assert_allclose(res_sin, self.sig_sin, atol=1e-06)

        assert_allclose(res_slope_off, self.sig_slope_mean, atol=1e-06)
        assert_allclose(res_sin_rand, self.sig_sin_rand, atol=1e-06)

        assert_allclose(res_rand_off, self.sig_rand, atol=1e-06)
        assert_allclose(res_sin_off, self.sig_sin, atol=1e-06)
        assert_allclose(res_sin_rand_off, self.sig_sin_rand, atol=1e-06)

        assert_allclose(res_rand_slope, self.sig_rand_slope_mean, atol=1e-06)
        assert_allclose(res_sin_slope, self.sig_sin_slope_mean, atol=1e-06)
        assert_allclose(res_sin_rand_slope, self.sig_sin_rand_slope_mean,
                        atol=1e-08)

        assert_allclose(res_rand_slope_off, self.sig_rand_slope_mean,
                        atol=1e-08)
        assert_allclose(res_sin_slope_off, self.sig_sin_slope_mean, atol=1e-06)
        assert_allclose(res_sin_rand_slope_off, self.sig_sin_rand_slope_mean,
                        atol=1e-08)

    def test_detrend_linear(self):
        '''test mlab.detrend_none'''
        res_off = mlab.detrend_linear(self.sig_off)
        res_slope = mlab.detrend_linear(self.sig_slope)

        res_slope_off = mlab.detrend_linear(self.sig_slope_off)

        assert_allclose(res_off, self.sig_zeros, atol=1e-06)
        assert_allclose(res_slope, self.sig_zeros, atol=1e-06)

        assert_allclose(res_slope_off, self.sig_zeros, atol=1e-06)


class spectral_testcase(unittest.TestCase):
    '''Check spectrum-related functions'''
    def setUp(self):
        '''shared set-up code for spectral tests'''
        self.Fs = 100.

        fstims = [self.Fs/4, self.Fs/5, self.Fs/10]
        x = np.arange(0, 10000, 1/self.Fs)

        self.NFFT = 1000*int(1/min(fstims) * self.Fs)
        self.nover = int(self.NFFT/2)
        self.pad_to = int(2**np.ceil(np.log2(self.NFFT)))

        # frequencies for specgram, psd, and csd
        freqss = np.linspace(0, self.Fs/2, num=self.pad_to//2+1)
        freqsd = np.linspace(-self.Fs/2, self.Fs/2, num=self.pad_to,
                             endpoint=False)

        # frequencies for complex, magnitude, angle, and phase spectrums
        freqssigs = np.linspace(0, self.Fs/2, num=len(x)//2+1)
        freqssigd = np.linspace(-self.Fs/2, self.Fs/2, num=len(x),
                                endpoint=False)

        # frequencies for complex, magnitude, angle, and phase spectrums
        freqs2xs = np.linspace(0, self.Fs/2, num=(2 * len(x))//2+1)
        freqs2xd = np.linspace(-self.Fs/2, self.Fs/2, num=2 * len(x),
                               endpoint=False)

        # frequencies for complex, magnitude, angle, and phase spectrums
        freqshalfs = np.linspace(0, self.Fs/2, num=(len(x)//2)//2+1)
        freqshalfd = np.linspace(-self.Fs/2, self.Fs/2, num=len(x)//2,
                                 endpoint=False)

        # frequencies for specgram, psd, and csd when NFFT and pad_to is None
        freqsnones = np.linspace(0, self.Fs/2, num=256//2+1)
        freqsnoned = np.linspace(-self.Fs/2, self.Fs/2, num=256,
                                 endpoint=False)

        # time points for specgram
        self.t = x[self.NFFT//2::self.NFFT-self.nover]
        self.tnone = x[256//2::256-0]

        # actual signals
        ytemp = np.zeros_like(x)
        yreal = [ytemp]
        ycomp = [ytemp.astype('complex')]
        for i, fstim in enumerate(fstims):
            ytemp = np.sin(fstim * x * np.pi * 2) * 10**i
            yreal.append(ytemp)
            ycomp.append(ytemp.astype('complex'))
        yreal.append(np.sum(yreal, axis=0))
        ycomp.append(np.sum(ycomp, axis=0))
        self.y = [yreal, ycomp]

        # get the list of frequencies in each test
        self.fstimsall = [[]] + [[f] for f in fstims] + [fstims]

        # figure out which freqs correspond to which sides value
        self.sides = ['default', 'onesided', 'twosided'] * 2

        freqsreal = [freqss, freqss, freqsd]
        freqscomplex = [freqsd, freqss, freqsd]
        self.freqs = [freqsreal, freqscomplex]

        freqssigreal = [freqssigs, freqssigs, freqssigd]
        freqssigcomplex = [freqssigd, freqssigs, freqssigd]
        self.freqssig = [freqssigreal, freqssigcomplex]

        freqs2xreal = [freqs2xs, freqs2xs, freqs2xd]
        freqs2xcomplex = [freqs2xd, freqs2xs, freqs2xd]
        self.freqs2x = [freqs2xreal, freqs2xcomplex]

        freqshalfreal = [freqshalfs, freqshalfs, freqshalfd]
        freqshalfcomplex = [freqshalfd, freqshalfs, freqshalfd]
        self.freqshalf = [freqshalfreal, freqshalfcomplex]

        freqsnonereal = [freqsnones, freqsnones, freqsnoned]
        freqsnonecomplex = [freqsnoned, freqsnones, freqsnoned]
        self.freqsnone = [freqsnonereal, freqsnonecomplex]

    def check_freqs(self, vals, freqs, resfreqs, fstims):
        '''Check the frequency values'''
        assert_allclose(resfreqs, freqs, atol=1e-06)
        for fstim in fstims:
            i = np.abs(resfreqs - fstim).argmin()
            self.assertTrue(vals[i] > vals[i+1])
            self.assertTrue(vals[i] > vals[i-1])

    def check_maxfreq(self, spec, fsp, fstims):
        '''Check that the peaks are correct'''
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
            self.assertAlmostEqual(maxfreq, fstimst[-1])
            del fstimst[-1]
            spect[maxind-50:maxind+50] = 0

    def test_spectral_helper_errors(self):
        '''test to make sure mlab._spectral_helper fails properly'''
        y = self.y[0][0]

        # test that mode 'complex' cannot be used if x is not y
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y+1, mode='complex')

        # test that mode 'magnitude' cannot be used if x is not y
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y+1, mode='magnitude')

        # test that mode 'angle' cannot be used if x is not y
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y+1, mode='angle')

        # test that mode 'phase' cannot be used if x is not y
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y+1, mode='phase')

        # test that unknown value for mode cannot be used
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, mode='spam')

        # test that unknown value for sides cannot be used
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y, sides='eggs')

        # test that noverlap cannot be larger than NFFT
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y, NFFT=10,
                                                 noverlap=20)

        # test that noverlap cannot be equal to NFFT
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, NFFT=10, noverlap=10)

        # test that the window length cannot be different from NFFT
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, y=y, NFFT=10,
                                                 window=np.ones(9))

        # test that NFFT cannot be odd
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._spectral_helper(x=y, NFFT=11)

        # test that mode 'default' cannot be used with _single_spectrum_helper
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._single_spectrum_helper(x=y, mode='default')

        # test that mode 'density' cannot be used with _single_spectrum_helper
        with self.assertRaises(ValueError):
            spec, fsp, t = mlab._single_spectrum_helper(x=y, mode='density')

    def test_spectral_helper_density(self):
        '''test mlab._spectral_helper in density mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab._spectral_helper(x=y, y=y,
                                                         NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='density')
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

    def test_spectral_helper_magnitude_specgram(self):
        '''test mlab._spectral_helper in magnitude mode with arguments
        used in specgram'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab._spectral_helper(x=y, y=y,
                                                         NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='magnitude')
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

    def test_spectral_helper_magnitude_magnitude_spectrum(self):
        '''test mlab._spectral_helper in magnitude mode with arguments
        used in magnitude_spectrum'''
        for ys, freqss in zip(self.y, self.freqssig):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab._spectral_helper(x=y, y=y,
                                                         NFFT=len(y),
                                                         Fs=self.Fs,
                                                         noverlap=0,
                                                         pad_to=len(y),
                                                         sides=side,
                                                         mode='magnitude')
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], 1)

    def test_csd(self):
        '''test mlab.csd'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.csd(y, y+1, NFFT=self.NFFT,
                                         Fs=self.Fs,
                                         noverlap=self.nover,
                                         pad_to=self.pad_to,
                                         sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_csd_nones(self):
        '''test mlab.csd when NFFT and pad_to are None'''
        for ys, freqss in zip(self.y, self.freqsnone):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.csd(y, y+1, NFFT=None,
                                         Fs=self.Fs,
                                         noverlap=0,
                                         pad_to=None,
                                         sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_csd_longer(self):
        '''test mlab.csd with NFFT longer than len(y)'''
        for ys, freqss in zip(self.y, self.freqsnone):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.csd(y[:250], y[:250]+1, NFFT=None,
                                         Fs=self.Fs,
                                         noverlap=0,
                                         pad_to=None,
                                         sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_psd(self):
        '''test mlab.psd'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.psd(y, NFFT=self.NFFT,
                                         Fs=self.Fs,
                                         noverlap=self.nover,
                                         pad_to=self.pad_to,
                                         sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)

    def test_psd_nones(self):
        '''test mlab.psd when NFFT and pad_to are None'''
        for ys, freqss in zip(self.y, self.freqsnone):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.psd(y, NFFT=None,
                                         Fs=self.Fs,
                                         noverlap=0,
                                         pad_to=None,
                                         sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)

    def test_psd_windowarray(self):
        '''test mlab.psd when window is an array'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.psd(y, NFFT=self.NFFT,
                                         Fs=self.Fs,
                                         noverlap=self.nover,
                                         pad_to=self.pad_to,
                                         sides=side,
                                         window=np.ones(self.NFFT))
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)

    def test_complex_spectrum(self):
        '''test mlab.complex_spectrum'''
        for ys, freqss in zip(self.y, self.freqssig):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.complex_spectrum(y, Fs=self.Fs,
                                                      sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_magnitude_spectrum(self):
        '''test mlab.magnitude_spectrum'''
        for ys, freqss in zip(self.y, self.freqssig):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.magnitude_spectrum(y, Fs=self.Fs,
                                                        sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)
                    self.check_maxfreq(spec, fsp, fstims)

    def test_magnitude_spectrum_double(self):
        '''test mlab.magnitude_spectrum with pad_to twice default'''
        for ys, freqss in zip(self.y, self.freqs2x):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.magnitude_spectrum(y, Fs=self.Fs,
                                                        sides=side,
                                                        pad_to=len(y) * 2)
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)
                    self.check_maxfreq(spec, fsp, fstims)

    def test_magnitude_spectrum_half(self):
        '''test mlab.magnitude_spectrum with pad_to half default'''
        for ys, freqss in zip(self.y, self.freqshalf):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.magnitude_spectrum(y, Fs=self.Fs,
                                                        sides=side,
                                                        pad_to=len(y) // 2)
                    self.assertEqual(spec.shape, freqs.shape)
                    self.check_freqs(spec, freqs, fsp, fstims)
                    self.check_maxfreq(spec, fsp, fstims)

    def test_angle_spectrum(self):
        '''test mlab.angle_spectrum'''
        for ys, freqss in zip(self.y, self.freqssig):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.angle_spectrum(y, Fs=self.Fs,
                                                    sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_phase_spectrum(self):
        '''test mlab.phase_spectrum'''
        for ys, freqss in zip(self.y, self.freqssig):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp = mlab.phase_spectrum(y, Fs=self.Fs,
                                                    sides=side)
                    self.assertEqual(spec.shape, freqs.shape)
                    assert_allclose(fsp, freqs, atol=1e-06)

    def test_specgram_auto(self):
        '''test mlab.specgram with no specified mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side)
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)
                    # since we are using a single freq, all time slices
                    # should be about the same
                    assert_allclose(np.diff(spec, axis=1).max(), 0, atol=1e-05)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_default(self):
        '''test mlab.specgram in default mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='default')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)
                    # since we are using a single freq, all time slices
                    # should be about the same
                    assert_allclose(np.diff(spec, axis=1).max(), 0, atol=1e-04)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_density(self):
        '''test mlab.specgram in density mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='density')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)
                    # since we are using a single freq, all time slices
                    # should be about the same
                    assert_allclose(np.diff(spec, axis=1).max(), 0, atol=1e-04)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_density_none(self):
        '''test mlab.specgram in density mode with NFFT and pad_to
        set to None'''
        for ys, freqss in zip(self.y, self.freqsnone):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=None,
                                                 Fs=self.Fs,
                                                 noverlap=0,
                                                 pad_to=None,
                                                 sides=side,
                                                 mode='density')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.tnone.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.tnone)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_complex(self):
        '''test mlab.specgram in complex mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='complex')
                    specm = np.mean(np.abs(spec), axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)
                    # since we are using a single freq, all time slices
                    # should be about the same
                    specr = spec.real
                    speci = spec.imag
                    assert_allclose(np.diff(specr, axis=1).max(), 0,
                                    atol=1e-05)
                    assert_allclose(np.diff(speci, axis=1).max(), 0,
                                    atol=1e-05)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_magnitude(self):
        '''test mlab.specgram in magnitude mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip([ys[1]], self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='magnitude')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)
                    # since we are using a single freq, all time slices
                    # should be about the same
                    assert_allclose(np.diff(spec, axis=1).max(), 0, atol=1e-06)
                    self.check_freqs(specm, freqs, fsp, fstims)

    def test_specgram_angle(self):
        '''test mlab.specgram in angle mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='angle')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)

    def test_specgram_phase(self):
        '''test mlab.specgram in phase mode'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    spec, fsp, t = mlab.specgram(y, NFFT=self.NFFT,
                                                 Fs=self.Fs,
                                                 noverlap=self.nover,
                                                 pad_to=self.pad_to,
                                                 sides=side,
                                                 mode='phase')
                    specm = np.mean(spec, axis=1)
                    self.assertEqual(spec.shape[0], freqs.shape[0])
                    self.assertEqual(spec.shape[1], self.t.shape[0])

                    assert_array_equal(fsp, freqs)
                    assert_array_equal(t, self.t)

    def test_psd_csd_equal(self):
        '''test that mlab.psd and mlab.csd are the same for x = y'''
        for ys in self.y:
            for y, fstims in zip(ys, self.fstimsall):
                for side in self.sides:
                    Pxx, freqsxx = mlab.psd(y, NFFT=self.NFFT,
                                            Fs=self.Fs,
                                            noverlap=self.nover,
                                            pad_to=self.pad_to,
                                            sides=side)
                    Pxy, freqsxy = mlab.csd(y, y, NFFT=self.NFFT,
                                            Fs=self.Fs,
                                            noverlap=self.nover,
                                            pad_to=self.pad_to,
                                            sides=side)
                    assert_array_equal(Pxx, Pxy)
                    assert_array_equal(freqsxx, freqsxy)

    def test_specgram_auto_default_density_equal(self):
        '''test that mlab.specgram without mode and with mode 'default' and
        'density' are all the same'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    speca, freqspeca, ta = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side)
                    specb, freqspecb, tb = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='default')
                    specc, freqspecc, tc = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='density')
                    assert_array_equal(speca, specb)
                    assert_array_equal(speca, specc)
                    assert_array_equal(freqspeca, freqspecb)
                    assert_array_equal(freqspeca, freqspecc)
                    assert_array_equal(ta, tb)
                    assert_array_equal(ta, tc)

    def test_specgram_complex_mag_angle_phase_equivalent(self):
        '''test that mlab.specgram with modes complex, magnitude, angle,
        and phase can be properly converted to one another'''
        for ys, freqss in zip(self.y, self.freqs):
            for y, fstims in zip(ys, self.fstimsall):
                for side, freqs in zip(self.sides, freqss):
                    specc, freqspecc, tc = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='complex')
                    specm, freqspecm, tm = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='magnitude')
                    speca, freqspeca, ta = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='angle')
                    specp, freqspecp, tp = mlab.specgram(y, NFFT=self.NFFT,
                                                         Fs=self.Fs,
                                                         noverlap=self.nover,
                                                         pad_to=self.pad_to,
                                                         sides=side,
                                                         mode='phase')

                    assert_array_equal(freqspecc, freqspecm)
                    assert_array_equal(freqspecc, freqspeca)
                    assert_array_equal(freqspecc, freqspecp)

                    assert_array_equal(tc, tm)
                    assert_array_equal(tc, ta)
                    assert_array_equal(tc, tp)

                    assert_allclose(np.abs(specc), specm, atol=1e-06)
                    assert_allclose(np.angle(specc), speca, atol=1e-06)
                    assert_allclose(np.unwrap(np.angle(specc), axis=0), specp,
                                    atol=1e-06)
                    assert_allclose(np.unwrap(speca, axis=0), specp,
                                    atol=1e-06)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
