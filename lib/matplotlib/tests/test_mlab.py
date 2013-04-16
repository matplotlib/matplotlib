from __future__ import division, print_function

import sys

import numpy as np
import matplotlib.mlab as mlab
import tempfile
import unittest


class general_test(unittest.TestCase):
    def test_colinear_pca(self):
        a = mlab.PCA._get_colinear()
        pca = mlab.PCA(a)

        np.testing.assert_allclose(pca.fracs[2:], 0., atol=1e-8)
        np.testing.assert_allclose(pca.Y[:, 2:], 0., atol=1e-8)

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
        np.testing.assert_allclose(expected, actual)

        # test scalar
        for pi, expectedi in zip(p, expected):
            actuali = mlab.prctile(ob1, pi)
            np.testing.assert_allclose(expectedi, actuali)


class csv_testcase(unittest.TestCase):
    def setUp(self):
        if sys.version_info[0] == 2:
            self.fd = tempfile.TemporaryFile(suffix='csv', mode="wb+")
        else:
            self.fd = tempfile.TemporaryFile(suffix='csv', mode="w+",
                                             newline='')

    def tearDown(self):
        self.fd.close()

    def test_recarray_csv_roundtrip(self):
        expected = np.recarray((99,),
                               [('x', np.float),
                                ('y', np.float),
                                ('t', np.float)])
        # initialising all values: uninitialised memory sometimes produces
        # floats that do not round-trip to string and back.
        expected['x'][:] = np.linspace(-1e9, -1, 99)
        expected['y'][:] = np.linspace(1, 1e9, 99)
        expected['t'][:] = np.linspace(0, 0.01, 99)

        mlab.rec2csv(expected, self.fd)
        self.fd.seek(0)
        actual = mlab.csv2rec(self.fd)

        np.testing.assert_allclose(expected['x'], actual['x'])
        np.testing.assert_allclose(expected['y'], actual['y'])
        np.testing.assert_allclose(expected['t'], actual['t'])

    def test_rec2csv_bad_shape_ValueError(self):
        bad = np.recarray((99, 4), [('x', np.float), ('y', np.float)])

        # the bad recarray should trigger a ValueError for having ndim > 1.
        self.assertRaises(ValueError, mlab.rec2csv, bad, self.fd)


class spectral_testcase(unittest.TestCase):
    def setUp(self):
        self.Fs = 100.

        self.fstims = [self.Fs/4, self.Fs/5, self.Fs/10]

        self.x = np.arange(0, 10000, 1/self.Fs)
        self.NFFT = 1000*int(1/min(self.fstims) * self.Fs)
        self.noverlap = int(self.NFFT/2)
        self.pad_to = int(2**np.ceil(np.log2(self.NFFT)))

        self.freqss = np.linspace(0, self.Fs/2, num=self.pad_to//2+1)
        self.freqsd = np.linspace(-self.Fs/2, self.Fs/2, num=self.pad_to,
                                  endpoint=False)

        self.t = self.x[self.NFFT//2::self.NFFT-self.noverlap]

        self.y = [np.zeros(self.x.size)]
        for i, fstim in enumerate(self.fstims):
            self.y.append(np.sin(fstim * self.x * np.pi * 2))
        self.y.append(np.sum(self.y, axis=0))

        # get the list of frequencies in each test
        self.fstimsall = [[]] + [[f] for f in self.fstims] + [self.fstims]

    def test_psd(self):
        for y, fstims in zip(self.y, self.fstimsall):
            Pxx1, freqs1 = mlab.psd(y, NFFT=self.NFFT,
                                    Fs=self.Fs,
                                    noverlap=self.noverlap,
                                    pad_to=self.pad_to,
                                    sides='default')
            np.testing.assert_array_equal(freqs1, self.freqss)
            for fstim in fstims:
                i = np.abs(freqs1 - fstim).argmin()
                self.assertTrue(Pxx1[i] > Pxx1[i+1])
                self.assertTrue(Pxx1[i] > Pxx1[i-1])

            Pxx2, freqs2 = mlab.psd(y, NFFT=self.NFFT,
                                    Fs=self.Fs,
                                    noverlap=self.noverlap,
                                    pad_to=self.pad_to,
                                    sides='onesided')
            np.testing.assert_array_equal(freqs2, self.freqss)
            for fstim in fstims:
                i = np.abs(freqs2 - fstim).argmin()
                self.assertTrue(Pxx2[i] > Pxx2[i+1])
                self.assertTrue(Pxx2[i] > Pxx2[i-1])

            Pxx3, freqs3 = mlab.psd(y, NFFT=self.NFFT,
                                    Fs=self.Fs,
                                    noverlap=self.noverlap,
                                    pad_to=self.pad_to,
                                    sides='twosided')
            np.testing.assert_array_equal(freqs3, self.freqsd)
            for fstim in fstims:
                i = np.abs(freqs3 - fstim).argmin()
                self.assertTrue(Pxx3[i] > Pxx3[i+1])
                self.assertTrue(Pxx3[i] > Pxx3[i-1])

    def test_specgram(self):
        for y, fstims in zip(self.y, self.fstimsall):
            Pxx1, freqs1, t1 = mlab.specgram(y, NFFT=self.NFFT,
                                             Fs=self.Fs,
                                             noverlap=self.noverlap,
                                             pad_to=self.pad_to,
                                             sides='default')
            Pxx1m = np.mean(Pxx1, axis=1)
            np.testing.assert_array_equal(freqs1, self.freqss)
            np.testing.assert_array_equal(t1, self.t)
            # since we are using a single freq, all time slices should be
            # about the same
            np.testing.assert_allclose(np.diff(Pxx1, axis=1).max(), 0,
                                       atol=1e-08)
            for fstim in fstims:
                i = np.abs(freqs1 - fstim).argmin()
                self.assertTrue(Pxx1m[i] > Pxx1m[i+1])
                self.assertTrue(Pxx1m[i] > Pxx1m[i-1])

            Pxx2, freqs2, t2 = mlab.specgram(y, NFFT=self.NFFT,
                                             Fs=self.Fs,
                                             noverlap=self.noverlap,
                                             pad_to=self.pad_to,
                                             sides='onesided')
            Pxx2m = np.mean(Pxx2, axis=1)
            np.testing.assert_array_equal(freqs2, self.freqss)
            np.testing.assert_array_equal(t2, self.t)
            np.testing.assert_allclose(np.diff(Pxx2, axis=1).max(), 0,
                                       atol=1e-08)
            for fstim in fstims:
                i = np.abs(freqs2 - fstim).argmin()
                self.assertTrue(Pxx2m[i] > Pxx2m[i+1])
                self.assertTrue(Pxx2m[i] > Pxx2m[i-1])

            Pxx3, freqs3, t3 = mlab.specgram(y, NFFT=self.NFFT,
                                             Fs=self.Fs,
                                             noverlap=self.noverlap,
                                             pad_to=self.pad_to,
                                             sides='twosided')
            Pxx3m = np.mean(Pxx3, axis=1)
            np.testing.assert_array_equal(freqs3, self.freqsd)
            np.testing.assert_array_equal(t3, self.t)
            np.testing.assert_allclose(np.diff(Pxx3, axis=1).max(), 0,
                                       atol=1e-08)
            for fstim in fstims:
                i = np.abs(freqs3 - fstim).argmin()
                self.assertTrue(Pxx3m[i] > Pxx3m[i+1])
                self.assertTrue(Pxx3m[i] > Pxx3m[i-1])


if __name__ == '__main__':
    unittest.main()
