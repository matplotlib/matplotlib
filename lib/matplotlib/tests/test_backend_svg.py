from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
from io import BytesIO
import xml.parsers.expat

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib

needs_tex = knownfailureif(
    not matplotlib.checkdep_tex(),
    "This test needs a TeX installation")


@cleanup
def test_visibility():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, 4 * np.pi, 50)
    y = np.sin(x)
    yerr = np.ones_like(y)

    a, b, c = ax.errorbar(x, y, yerr=yerr, fmt='ko')
    for artist in b:
        artist.set_visible(False)

    fd = BytesIO()
    fig.savefig(fd, format='svg')

    fd.seek(0)
    buf = fd.read()
    fd.close()

    parser = xml.parsers.expat.ParserCreate()
    parser.Parse(buf)  # this will raise ExpatError if the svg is invalid


@image_comparison(baseline_images=['fill_black_with_alpha'], remove_text=True,
                  extensions=['svg'])
def test_fill_black_with_alpha():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=[0, 0.1, 1], y=[0, 0, 0], c='k', alpha=0.1, s=10000)


@image_comparison(baseline_images=['noscale'], remove_text=True)
def test_noscale():
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(Z, cmap='gray', interpolation='none')


@cleanup
def test_composite_images():
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
    with BytesIO() as svg:
        fig.savefig(svg, format="svg")
        svg.seek(0)
        buff = svg.read()
        assert buff.count(six.b('<image ')) == 1
    plt.rcParams['image.composite_image'] = False
    with BytesIO() as svg:
        fig.savefig(svg, format="svg")
        svg.seek(0)
        buff = svg.read()
        assert buff.count(six.b('<image ')) == 2


@cleanup
def test_text_urls():
    fig = plt.figure()

    test_url = "http://test_text_urls.matplotlib.org"
    fig.suptitle("test_text_urls", url=test_url)

    fd = BytesIO()
    fig.savefig(fd, format='svg')
    fd.seek(0)
    buf = fd.read().decode()
    fd.close()

    expected = '<a xlink:href="{0}">'.format(test_url)
    assert expected in buf


@image_comparison(baseline_images=['bold_font_output'], extensions=['svg'])
def test_bold_font_output():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel('nonbold-xlabel')
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    ax.set_title('bold-title', fontweight='bold')


@image_comparison(baseline_images=['bold_font_output_with_none_fonttype'],
                  extensions=['svg'])
def test_bold_font_output_with_none_fonttype():
    plt.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel('nonbold-xlabel')
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    ax.set_title('bold-title', fontweight='bold')


def _test_determinism_save(filename, usetex):
    # This function is mostly copy&paste from "def test_visibility"
    # To require no GUI, we use Figure and FigureCanvasSVG
    # instead of plt.figure and fig.savefig
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib import rc
    rc('svg', hashsalt='asdf')
    rc('text', usetex=usetex)

    fig = Figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, 4 * np.pi, 50)
    y = np.sin(x)
    yerr = np.ones_like(y)

    a, b, c = ax.errorbar(x, y, yerr=yerr, fmt='ko')
    for artist in b:
        artist.set_visible(False)
    ax.set_title('A string $1+2+\sigma$')
    ax.set_xlabel('A string $1+2+\sigma$')
    ax.set_ylabel('A string $1+2+\sigma$')

    FigureCanvasSVG(fig).print_svg(filename)


def _test_determinism(filename, usetex):
    import os
    import sys
    from subprocess import check_call
    from nose.tools import assert_equal
    plots = []
    for i in range(3):
        check_call([sys.executable, '-R', '-c',
                    'import matplotlib; '
                    'matplotlib.use("svg"); '
                    'from matplotlib.tests.test_backend_svg '
                    'import _test_determinism_save;'
                    '_test_determinism_save(%r, %r)' % (filename, usetex)])
        with open(filename, 'rb') as fd:
            plots.append(fd.read())
        os.unlink(filename)
    for p in plots[1:]:
        assert_equal(p, plots[0])


@cleanup
def test_determinism_notex():
    # unique filename to allow for parallel testing
    _test_determinism('determinism_notex.svg', usetex=False)


@cleanup
@needs_tex
def test_determinism_tex():
    # unique filename to allow for parallel testing
    _test_determinism('determinism_tex.svg', usetex=True)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
