# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import re

import six

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup, knownfailureif


needs_ghostscript = knownfailureif(
    matplotlib.checkdep_ghostscript()[0] is None,
    "This test needs a ghostscript installation")


needs_tex = knownfailureif(
    not matplotlib.checkdep_tex(),
    "This test needs a TeX installation")


def _test_savefig_to_stringio(format='ps'):
    buffers = [
        six.moves.StringIO(),
        io.StringIO(),
        io.BytesIO()]

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("Déjà vu")
    for buffer in buffers:
        plt.savefig(buffer, format=format)

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
def test_savefig_to_stringio_with_usetex():
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['text.usetex'] = True
    _test_savefig_to_stringio()


@cleanup
def test_savefig_to_stringio_eps():
    _test_savefig_to_stringio(format='eps')


@cleanup
@needs_tex
def test_savefig_to_stringio_with_usetex_eps():
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['text.usetex'] = True
    _test_savefig_to_stringio(format='eps')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
