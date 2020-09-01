import datetime
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images, ImageComparisonFailure
from matplotlib.testing.decorators import image_comparison, _image_directories
from matplotlib.backends.backend_pgf import PdfPages, common_texification

baseline_dir, result_dir = _image_directories(lambda: 'dummy func')


def check_for(texsystem):
    with TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir, "test.tex")
        tex_path.write_text(r"""
            \documentclass{minimal}
            \usepackage{pgf}
            \begin{document}
            \typeout{pgfversion=\pgfversion}
            \makeatletter
            \@@end
        """)
        try:
            subprocess.check_call(
                [texsystem, "-halt-on-error", str(tex_path)], cwd=tmpdir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return False
        return True


needs_xelatex = pytest.mark.skipif(not check_for('xelatex'),
                                   reason='xelatex + pgf is required')
needs_pdflatex = pytest.mark.skipif(not check_for('pdflatex'),
                                    reason='pdflatex + pgf is required')
needs_lualatex = pytest.mark.skipif(not check_for('lualatex'),
                                    reason='lualatex + pgf is required')


def _has_sfmath():
    return (shutil.which("kpsewhich")
            and subprocess.run(["kpsewhich", "sfmath.sty"],
                               stdout=subprocess.PIPE).returncode == 0)


def compare_figure(fname, savefig_kwargs={}, tol=0):
    actual = os.path.join(result_dir, fname)
    plt.savefig(actual, **savefig_kwargs)

    expected = os.path.join(result_dir, "expected_%s" % fname)
    shutil.copyfile(os.path.join(baseline_dir, fname), expected)
    err = compare_images(expected, actual, tol=tol)
    if err:
        raise ImageComparisonFailure(err)


def create_figure():
    plt.figure()
    x = np.linspace(0, 1, 15)

    # line plot
    plt.plot(x, x ** 2, "b-")

    # marker
    plt.plot(x, 1 - x**2, "g>")

    # filled paths and patterns
    plt.fill_between([0., .4], [.4, 0.], hatch='//', facecolor="lightgray",
                     edgecolor="red")
    plt.fill([3, 3, .8, .8, 3], [2, -2, -2, 0, 2], "b")

    # text and typesetting
    plt.plot([0.9], [0.5], "ro", markersize=3)
    plt.text(0.9, 0.5, 'unicode (ü, °, µ) and math ($\\mu_i = x_i^2$)',
             ha='right', fontsize=20)
    plt.ylabel('sans-serif, blue, $\\frac{\\sqrt{x}}{y^2}$..',
               family='sans-serif', color='blue')

    plt.xlim(0, 1)
    plt.ylim(0, 1)


@pytest.mark.parametrize('plain_text, escaped_text', [
    (r'quad_sum: $\sum x_i^2$', r'quad\_sum: \(\displaystyle \sum x_i^2\)'),
    (r'no \$splits \$ here', r'no \$splits \$ here'),
    ('with_underscores', r'with\_underscores'),
    ('% not a comment', r'\% not a comment'),
    ('^not', r'\^not'),
])
def test_common_texification(plain_text, escaped_text):
    assert common_texification(plain_text) == escaped_text


# test compiling a figure to pdf with xelatex
@needs_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_xelatex.pdf'], style='default')
def test_xelatex():
    rc_xelatex = {'font.family': 'serif',
                  'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)
    create_figure()


# test compiling a figure to pdf with pdflatex
@needs_pdflatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_pdflatex.pdf'], style='default')
def test_pdflatex():
    if os.environ.get('APPVEYOR', False):
        pytest.xfail("pdflatex test does not work on appveyor due to missing "
                     "LaTeX fonts")

    rc_pdflatex = {'font.family': 'serif',
                   'pgf.rcfonts': False,
                   'pgf.texsystem': 'pdflatex',
                   'pgf.preamble': ('\\usepackage[utf8x]{inputenc}'
                                    '\\usepackage[T1]{fontenc}')}
    mpl.rcParams.update(rc_pdflatex)
    create_figure()


# test updating the rc parameters for each figure
@needs_xelatex
@needs_pdflatex
@pytest.mark.skipif(not _has_sfmath(), reason='needs sfmath.sty')
@pytest.mark.style('default')
@pytest.mark.backend('pgf')
def test_rcupdate():
    rc_sets = [{'font.family': 'sans-serif',
                'font.size': 30,
                'figure.subplot.left': .2,
                'lines.markersize': 10,
                'pgf.rcfonts': False,
                'pgf.texsystem': 'xelatex'},
               {'font.family': 'monospace',
                'font.size': 10,
                'figure.subplot.left': .1,
                'lines.markersize': 20,
                'pgf.rcfonts': False,
                'pgf.texsystem': 'pdflatex',
                'pgf.preamble': ('\\usepackage[utf8x]{inputenc}'
                                 '\\usepackage[T1]{fontenc}'
                                 '\\usepackage{sfmath}')}]
    tol = [6, 0]
    for i, rc_set in enumerate(rc_sets):
        with mpl.rc_context(rc_set):
            create_figure()
            compare_figure('pgf_rcupdate%d.pdf' % (i + 1), tol=tol[i])


# test backend-side clipping, since large numbers are not supported by TeX
@needs_xelatex
@pytest.mark.style('default')
@pytest.mark.backend('pgf')
def test_pathclip():
    rc_xelatex = {'font.family': 'serif',
                  'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)

    plt.figure()
    plt.plot([0., 1e100], [0., 1e100])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # this test passes if compiling/saving to pdf works (no image comparison)
    plt.savefig(os.path.join(result_dir, "pgf_pathclip.pdf"))


# test mixed mode rendering
@needs_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_mixedmode.pdf'], style='default')
def test_mixedmode():
    rc_xelatex = {'font.family': 'serif',
                  'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)

    Y, X = np.ogrid[-1:1:40j, -1:1:40j]
    plt.figure()
    plt.pcolor(X**2 + Y**2).set_rasterized(True)


# test bbox_inches clipping
@needs_xelatex
@pytest.mark.style('default')
@pytest.mark.backend('pgf')
def test_bbox_inches():
    rc_xelatex = {'font.family': 'serif',
                  'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)

    Y, X = np.ogrid[-1:1:40j, -1:1:40j]
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(range(5))
    ax2 = fig.add_subplot(122)
    ax2.plot(range(5))
    plt.tight_layout()

    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    compare_figure('pgf_bbox_inches.pdf', savefig_kwargs={'bbox_inches': bbox},
                   tol=0)


@pytest.mark.style('default')
@pytest.mark.backend('pgf')
@pytest.mark.parametrize('system', [
    pytest.param('lualatex', marks=[needs_lualatex]),
    pytest.param('pdflatex', marks=[needs_pdflatex]),
    pytest.param('xelatex', marks=[needs_xelatex]),
])
def test_pdf_pages(system):
    rc_pdflatex = {
        'font.family': 'serif',
        'pgf.rcfonts': False,
        'pgf.texsystem': system,
    }
    mpl.rcParams.update(rc_pdflatex)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(5))
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(3, 2))
    ax2.plot(range(5))
    fig2.tight_layout()

    path = os.path.join(result_dir, f'pdfpages_{system}.pdf')
    md = {
        'Author': 'me',
        'Title': 'Multipage PDF with pgf',
        'Subject': 'Test page',
        'Keywords': 'test,pdf,multipage',
        'ModDate': datetime.datetime(
            1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),
        'Trapped': 'Unknown'
    }

    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig1)

        assert pdf.get_pagecount() == 3


@pytest.mark.style('default')
@pytest.mark.backend('pgf')
@pytest.mark.parametrize('system', [
    pytest.param('lualatex', marks=[needs_lualatex]),
    pytest.param('pdflatex', marks=[needs_pdflatex]),
    pytest.param('xelatex', marks=[needs_xelatex]),
])
def test_pdf_pages_metadata_check(monkeypatch, system):
    # Basically the same as test_pdf_pages, but we keep it separate to leave
    # pikepdf as an optional dependency.
    pikepdf = pytest.importorskip('pikepdf')
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')

    mpl.rcParams.update({'pgf.texsystem': system})

    fig, ax = plt.subplots()
    ax.plot(range(5))

    md = {
        'Author': 'me',
        'Title': 'Multipage PDF with pgf',
        'Subject': 'Test page',
        'Keywords': 'test,pdf,multipage',
        'ModDate': datetime.datetime(
            1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),
        'Trapped': 'True'
    }
    path = os.path.join(result_dir, f'pdfpages_meta_check_{system}.pdf')
    with PdfPages(path, metadata=md) as pdf:
        pdf.savefig(fig)

    with pikepdf.Pdf.open(path) as pdf:
        info = {k: str(v) for k, v in pdf.docinfo.items()}

    # Not set by us, so don't bother checking.
    if '/PTEX.FullBanner' in info:
        del info['/PTEX.FullBanner']
    if '/PTEX.Fullbanner' in info:
        del info['/PTEX.Fullbanner']

    assert info == {
        '/Author': 'me',
        '/CreationDate': 'D:19700101000000Z',
        '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org',
        '/Keywords': 'test,pdf,multipage',
        '/ModDate': 'D:19680801000000Z',
        '/Producer': f'Matplotlib pgf backend v{mpl.__version__}',
        '/Subject': 'Test page',
        '/Title': 'Multipage PDF with pgf',
        '/Trapped': '/True',
    }


@needs_xelatex
def test_tex_restart_after_error():
    fig = plt.figure()
    fig.suptitle(r"\oops")
    with pytest.raises(ValueError):
        fig.savefig(BytesIO(), format="pgf")

    fig = plt.figure()  # start from scratch
    fig.suptitle(r"this is ok")
    fig.savefig(BytesIO(), format="pgf")


@needs_xelatex
def test_bbox_inches_tight(tmpdir):
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])
    fig.savefig(os.path.join(tmpdir, "test.pdf"), backend="pgf",
                bbox_inches="tight")
