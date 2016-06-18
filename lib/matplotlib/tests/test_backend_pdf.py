# -*- encoding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import io
import os

import numpy as np
from matplotlib import cm, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import (image_comparison, knownfailureif,
                                           cleanup)

if 'TRAVIS' not in os.environ:
    @image_comparison(baseline_images=['pdf_use14corefonts'],
                      extensions=['pdf'])
    def test_use14corefonts():
        rcParams['pdf.use14corefonts'] = True
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.size'] = 8
        rcParams['font.sans-serif'] = ['Helvetica']
        rcParams['pdf.compression'] = 0

        text = '''A three-line text positioned just above a blue line
    and containing some French characters and the euro symbol:
    "Merci pépé pour les 10 €"'''


@cleanup
def test_type42():
    rcParams['pdf.fonttype'] = 42

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3])
    fig.savefig(io.BytesIO())


@cleanup
def test_multipage_pagecount():
    with PdfPages(io.BytesIO()) as pdf:
        assert pdf.get_pagecount() == 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3])
        fig.savefig(pdf, format="pdf")
        assert pdf.get_pagecount() == 1
        pdf.savefig()
        assert pdf.get_pagecount() == 2


@cleanup
def test_multipage_keep_empty():
    from matplotlib.backends.backend_pdf import PdfPages
    from tempfile import NamedTemporaryFile
    # test empty pdf files
    # test that an empty pdf is left behind with keep_empty=True (default)
    with NamedTemporaryFile(delete=False) as tmp:
        with PdfPages(tmp) as pdf:
            filename = pdf._file.fh.name
        assert os.path.exists(filename)
    os.remove(filename)
    # test if an empty pdf is deleting itself afterwards with keep_empty=False
    with PdfPages(filename, keep_empty=False) as pdf:
        pass
    assert not os.path.exists(filename)
    # test pdf files with content, they should never be deleted
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3])
    # test that a non-empty pdf is left behind with keep_empty=True (default)
    with NamedTemporaryFile(delete=False) as tmp:
        with PdfPages(tmp) as pdf:
            filename = pdf._file.fh.name
            pdf.savefig()
        assert os.path.exists(filename)
    os.remove(filename)
    # test that a non-empty pdf is left behind with keep_empty=False
    with NamedTemporaryFile(delete=False) as tmp:
        with PdfPages(tmp, keep_empty=False) as pdf:
            filename = pdf._file.fh.name
            pdf.savefig()
        assert os.path.exists(filename)
    os.remove(filename)


@cleanup
def test_composite_image():
    #Test that figures can be saved with and without combining multiple images
    #(on a single set of axes) into a single composite image.
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 3)
    ax.imshow(Z, extent=[0, 1, 0, 1])
    ax.imshow(Z[::-1], extent=[2, 3, 0, 1])
    plt.rcParams['image.composite_image'] = True
    with PdfPages(io.BytesIO()) as pdf:
        fig.savefig(pdf, format="pdf")
        assert len(pdf._file._images.keys()) == 1
    plt.rcParams['image.composite_image'] = False
    with PdfPages(io.BytesIO()) as pdf:
        fig.savefig(pdf, format="pdf")
        assert len(pdf._file._images.keys()) == 2


@cleanup
def test_source_date_epoch():
    # Test SOURCE_DATE_EPOCH support
    try:
        # save current value of SOURCE_DATE_EPOCH
        sde = os.environ.pop('SOURCE_DATE_EPOCH',None)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = [1, 2, 3, 4, 5]
        ax.plot(x, x)
        os.environ['SOURCE_DATE_EPOCH'] = "946684800"
        with io.BytesIO() as pdf:
            fig.savefig(pdf, format="pdf")
            pdf.seek(0)
            buff = pdf.read()
            assert b"/CreationDate (D:20000101000000Z)" in buff
        os.environ.pop('SOURCE_DATE_EPOCH',None)
        with io.BytesIO() as pdf:
            fig.savefig(pdf, format="pdf")
            pdf.seek(0)
            buff = pdf.read()
            assert not b"/CreationDate (D:20000101000000Z)" in buff
    finally:
        # Restores SOURCE_DATE_EPOCH
        if sde == None:
            os.environ.pop('SOURCE_DATE_EPOCH',None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = sde


def _test_determinism_save(filename, objects=''):
    # save current value of SOURCE_DATE_EPOCH and set it
    # to a constant value, so that time difference is not
    # taken into account
    sde = os.environ.pop('SOURCE_DATE_EPOCH',None)
    os.environ['SOURCE_DATE_EPOCH'] = "946684800"

    fig = plt.figure()

    if 'm' in objects:
        # use different markers, to be recorded in the PdfFile object
        ax1 = fig.add_subplot(1, 6, 1)
        x = range(10)
        ax1.plot(x, [1] * 10, marker=u'D')
        ax1.plot(x, [2] * 10, marker=u'x')
        ax1.plot(x, [3] * 10, marker=u'^')
        ax1.plot(x, [4] * 10, marker=u'H')
        ax1.plot(x, [5] * 10, marker=u'v')

    if 'h' in objects:
        # also use different hatch patterns
        ax2 = fig.add_subplot(1, 6, 2)
        bars = ax2.bar(range(1, 5), range(1, 5)) + \
               ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5))
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

    if 'i' in objects:
        # also use different images
        A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
        A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
        fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
        A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
        fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')

    x=range(5)
    fig.add_subplot(1, 6, 6).plot(x,x)

    fig.savefig(filename, format="pdf")

    # Restores SOURCE_DATE_EPOCH
    if sde == None:
        os.environ.pop('SOURCE_DATE_EPOCH',None)
    else:
        os.environ['SOURCE_DATE_EPOCH'] = sde


def _test_determinism(objects=''):
    import sys
    from subprocess import check_call
    from nose.tools import assert_equal
    filename = 'determinism_O%s.pdf' % objects
    plots = []
    for i in range(3):
        check_call([sys.executable, '-R', '-c',
                    'import matplotlib; '
                    'matplotlib.use("pdf"); '
                    'from matplotlib.tests.test_backend_pdf '
                    'import _test_determinism_save;'
                    '_test_determinism_save(%r,%r)' % (filename,objects)])
        with open(filename, 'rb') as fd:
            plots.append(fd.read())
        os.unlink(filename)
    for p in plots[1:]:
        assert_equal(p,plots[0])


@cleanup
def test_determinism_plain():
    """Test for reproducible PDF output: simple figure"""
    _test_determinism()


@cleanup
def test_determinism_images():
    """Test for reproducible PDF output: figure with different images"""
    _test_determinism('i')


@cleanup
def test_determinism_hatches():
    """Test for reproducible PDF output: figure with different hatches"""
    _test_determinism('h')


@cleanup
def test_determinism_markers():
    """Test for reproducible PDF output: figure with different markers"""
    _test_determinism('m')


@cleanup
def test_determinism_all():
    """Test for reproducible PDF output"""
    _test_determinism('mhi')


@image_comparison(baseline_images=['hatching_legend'],
                  extensions=['pdf'])
def test_hatching_legend():
    """Test for correct hatching on patches in legend"""
    fig = plt.figure(figsize=(1, 2))

    a = plt.Rectangle([0, 0], 0, 0, facecolor="green", hatch="XXXX")
    b = plt.Rectangle([0, 0], 0, 0, facecolor="blue", hatch="XXXX")

    fig.legend([a, b, a, b], ["", "", "", ""])


@image_comparison(baseline_images=['grayscale_alpha'],
                  extensions=['pdf'])
def test_grayscale_alpha():
    """Masking images with NaN did not work for grayscale images"""
    x, y = np.ogrid[-2:2:.1, -2:2:.1]
    dd = np.exp(-(x**2 + y**2))
    dd[dd < .1] = np.nan
    fig, ax = plt.subplots()
    ax.imshow(dd, interpolation='none', cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])
