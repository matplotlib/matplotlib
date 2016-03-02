# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import re
import numpy as np
from matplotlib.externals import six

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.testing.decorators import cleanup, knownfailureif


needs_ghostscript = knownfailureif(
    matplotlib.checkdep_ghostscript()[0] is None,
    "This test needs a ghostscript installation")


needs_tex = knownfailureif(
    not matplotlib.checkdep_tex(),
    "This test needs a TeX installation")


def _test_savefig_to_stringio(format='ps', use_log=False):
    fig, ax = plt.subplots()
    buffers = [
        six.moves.StringIO(),
        io.StringIO(),
        io.BytesIO()]

    if use_log:
        ax.set_yscale('log')

    ax.plot([1, 2], [1, 2])
    ax.set_title("Déjà vu")
    for buffer in buffers:
        fig.savefig(buffer, format=format)

    values = [x.getvalue() for x in buffers]

    if six.PY3:
        values = [
            values[0].encode('ascii'),
            values[1].encode('ascii'),
            values[2]]

    # Remove comments from the output.  This includes things that
    # could change from run to run, such as the time.
    values = [re.sub(b'%%.*?\n', b'', x) for x in values]

    assert values[0] == values[1]
    assert values[1] == values[2].replace(b'\r\n', b'\n')
    for buffer in buffers:
        buffer.close()


@cleanup
def test_savefig_to_stringio():
    _test_savefig_to_stringio()


@cleanup
@needs_ghostscript
def test_savefig_to_stringio_with_distiller():
    matplotlib.rcParams['ps.usedistiller'] = 'ghostscript'
    _test_savefig_to_stringio()


@cleanup
@needs_tex
@needs_ghostscript
def test_savefig_to_stringio_with_usetex():
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['text.usetex'] = True
    _test_savefig_to_stringio()


@cleanup
def test_savefig_to_stringio_eps():
    _test_savefig_to_stringio(format='eps')


@cleanup
def test_savefig_to_stringio_eps_afm():
    matplotlib.rcParams['ps.useafm'] = True
    _test_savefig_to_stringio(format='eps', use_log=True)


@cleanup
@needs_tex
@needs_ghostscript
def test_savefig_to_stringio_with_usetex_eps():
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['text.usetex'] = True
    _test_savefig_to_stringio(format='eps')


@cleanup
def test_composite_image():
    # Test that figures can be saved with and without combining multiple images
    # (on a single set of axes) into a single composite image.
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 3)
    ax.imshow(Z, extent=[0, 1, 0, 1])
    ax.imshow(Z[::-1], extent=[2, 3, 0, 1])
    plt.rcParams['image.composite_image'] = True
    with io.BytesIO() as ps:
        fig.savefig(ps, format="ps")
        ps.seek(0)
        buff = ps.read()
        assert buff.count(six.b(' colorimage')) == 1
    plt.rcParams['image.composite_image'] = False
    with io.BytesIO() as ps:
        fig.savefig(ps, format="ps")
        ps.seek(0)
        buff = ps.read()
        assert buff.count(six.b(' colorimage')) == 2


@cleanup
def test_patheffects():
    with matplotlib.rc_context():
        matplotlib.rcParams['path.effects'] = [
            patheffects.withStroke(linewidth=4, foreground='w')]
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with io.BytesIO() as ps:
            fig.savefig(ps, format='ps')


@cleanup
@needs_tex
@needs_ghostscript
def test_tilde_in_tempfilename():
    # Tilde ~ in the tempdir path (e.g. TMPDIR, TMP oder TEMP on windows
    # when the username is very long and windows uses a short name) breaks
    # latex before https://github.com/matplotlib/matplotlib/pull/5928
    import tempfile
    import shutil
    import os
    import os.path

    tempdir = None
    old_tempdir = tempfile.tempdir
    try:
        # change the path for new tempdirs, which is used
        # internally by the ps backend to write a file
        tempdir = tempfile.mkdtemp()
        base_tempdir = os.path.join(tempdir, "short~1")
        os.makedirs(base_tempdir)
        tempfile.tempdir = base_tempdir

        # usetex results in the latex call, which does not like the ~
        plt.rc('text', usetex=True)
        plt.plot([1, 2, 3, 4])
        plt.xlabel(r'\textbf{time} (s)')
        #matplotlib.verbose.set_level("debug")
        output_eps = os.path.join(base_tempdir, 'tex_demo.eps')
        # use the PS backend to write the file...
        plt.savefig(output_eps, format="ps")
    finally:
        tempfile.tempdir = old_tempdir
        if tempdir:
            try:
                shutil.rmtree(tempdir)
            except Exception as e:
                # do not break if this is not removeable...
                print(e)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
