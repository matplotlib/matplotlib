import io
import os
from pathlib import Path
import re
import tempfile

import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cbook, patheffects
from matplotlib.testing.decorators import image_comparison


needs_ghostscript = pytest.mark.skipif(
    "eps" not in mpl.testing.compare.converter,
    reason="This test needs a ghostscript installation")
needs_usetex = pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")


# This tests tends to hit a TeX cache lock on AppVeyor.
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('orientation', ['portrait', 'landscape'])
@pytest.mark.parametrize('format, use_log, rcParams', [
    ('ps', False, {}),
    pytest.param('ps', False, {'ps.usedistiller': 'ghostscript'},
                 marks=needs_ghostscript),
    pytest.param('ps', False, {'text.usetex': True},
                 marks=[needs_ghostscript, needs_usetex]),
    ('eps', False, {}),
    ('eps', True, {'ps.useafm': True}),
    pytest.param('eps', False, {'text.usetex': True},
                 marks=[needs_ghostscript, needs_usetex]),
], ids=[
    'ps',
    'ps with distiller',
    'ps with usetex',
    'eps',
    'eps afm',
    'eps with usetex'
])
def test_savefig_to_stringio(format, use_log, rcParams, orientation,
                             monkeypatch):
    mpl.rcParams.update(rcParams)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")  # For reproducibility.

    fig, ax = plt.subplots()

    with io.StringIO() as s_buf, io.BytesIO() as b_buf:

        if use_log:
            ax.set_yscale('log')

        ax.plot([1, 2], [1, 2])
        title = "Déjà vu"
        if not mpl.rcParams["text.usetex"]:
            title += " \N{MINUS SIGN}\N{EURO SIGN}"
        ax.set_title(title)
        fig.savefig(s_buf, format=format, orientation=orientation)
        fig.savefig(b_buf, format=format, orientation=orientation)

        s_val = s_buf.getvalue().encode('ascii')
        b_val = b_buf.getvalue()

        assert s_val == b_val.replace(b'\r\n', b'\n')


def test_patheffects():
    mpl.rcParams['path.effects'] = [
        patheffects.withStroke(linewidth=4, foreground='w')]
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with io.BytesIO() as ps:
        fig.savefig(ps, format='ps')


@needs_usetex
@needs_ghostscript
def test_tilde_in_tempfilename(tmpdir):
    # Tilde ~ in the tempdir path (e.g. TMPDIR, TMP or TEMP on windows
    # when the username is very long and windows uses a short name) breaks
    # latex before https://github.com/matplotlib/matplotlib/pull/5928
    base_tempdir = Path(tmpdir, "short-1")
    base_tempdir.mkdir()
    # Change the path for new tempdirs, which is used internally by the ps
    # backend to write a file.
    with cbook._setattr_cm(tempfile, tempdir=str(base_tempdir)):
        # usetex results in the latex call, which does not like the ~
        mpl.rcParams['text.usetex'] = True
        plt.plot([1, 2, 3, 4])
        plt.xlabel(r'\textbf{time} (s)')
        # use the PS backend to write the file...
        plt.savefig(base_tempdir / 'tex_demo.eps', format="ps")


@image_comparison(["empty.eps"])
def test_transparency():
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.plot([0, 1], color="r", alpha=0)
    ax.text(.5, .5, "foo", color="r", alpha=0)


@needs_usetex
def test_failing_latex(tmpdir):
    """Test failing latex subprocess call"""
    mpl.rcParams['text.usetex'] = True
    # This fails with "Double subscript"
    plt.xlabel("$22_2_2$")
    with pytest.raises(RuntimeError):
        plt.savefig(Path(tmpdir, "tmpoutput.ps"))
