# Author: Óscar Nájera
# License: 3-clause BSD
"""Test the SG pipeline used with Sphinx and tinybuild."""

import codecs
from io import StringIO
import os
import os.path as op
from pathlib import Path
import re
import shutil
import sys
import time
import glob
import json

from packaging.version import Version

from sphinx import __version__ as sphinx_version
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util.docutils import docutils_namespace
from sphinx_gallery.utils import (
    _get_image,
    scale_image,
    _has_optipng,
    _has_pypandoc,
    _has_graphviz,
)

import pytest

# file inventory for tinybuild:

# total number of plot_*.py files in tinybuild/examples + examples_rst_index
# + examples_with_rst
N_EXAMPLES = 15 + 3 + 2
N_FAILING = 2
N_GOOD = N_EXAMPLES - N_FAILING  # galleries that run w/o error
# passthroughs and non-executed examples in examples + examples_rst_index
# + examples_with_rst
N_PASS = 3 + 0 + 2
# indices SG generates  (extra non-plot*.py file)
# + examples_rst_index + examples_with_rst
N_INDEX = 2 + 1 + 3
# SG execution times (examples + examples_rst_index + examples_with_rst + root-level)
N_EXECUTE = 2 + 3 + 1 + 1
# gen_modules + sg_api_usage + doc/index.rst + minigallery.rst
N_OTHER = 9 + 1 + 1 + 1 + 1
N_RST = N_EXAMPLES + N_PASS + N_INDEX + N_EXECUTE + N_OTHER
N_RST = f"({N_RST}|{N_RST - 1}|{N_RST - 2})"  # AppVeyor weirdness


pytest.importorskip("jupyterlite_sphinx")  # needed for tinybuild


@pytest.fixture(scope="module")
def sphinx_app(tmpdir_factory, req_mpl, req_pil):
    return _sphinx_app(tmpdir_factory, "html")


@pytest.fixture(scope="module")
def sphinx_dirhtml_app(tmpdir_factory, req_mpl, req_pil):
    return _sphinx_app(tmpdir_factory, "dirhtml")


def _sphinx_app(tmpdir_factory, buildername):
    # Skip if numpy not installed
    pytest.importorskip("numpy")

    temp_dir = (tmpdir_factory.getbasetemp() / f"root_{buildername}").strpath
    src_dir = op.join(op.dirname(__file__), "tinybuild")

    def ignore(src, names):
        return ("_build", "gen_modules", "auto_examples")

    shutil.copytree(src_dir, temp_dir, ignore=ignore)
    # For testing iteration, you can get similar behavior just doing `make`
    # inside the tinybuild/doc directory
    conf_dir = op.join(temp_dir, "doc")
    out_dir = op.join(conf_dir, "_build", buildername)
    toctrees_dir = op.join(conf_dir, "_build", "toctrees")
    # Avoid warnings about re-registration, see:
    # https://github.com/sphinx-doc/sphinx/issues/5038
    with docutils_namespace():
        app = Sphinx(
            conf_dir,
            conf_dir,
            out_dir,
            toctrees_dir,
            buildername=buildername,
            status=StringIO(),
            warning=StringIO(),
        )
        # need to build within the context manager
        # for automodule and backrefs to work
        app.build(False, [])
    return app


def test_timings(sphinx_app):
    """Test that a timings page is created."""
    out_dir = sphinx_app.outdir
    src_dir = sphinx_app.srcdir

    # local folder
    timings_rst = op.join(src_dir, "auto_examples", "sg_execution_times.rst")
    assert op.isfile(timings_rst)
    with codecs.open(timings_rst, "r", "utf-8") as fid:
        content = fid.read()
    assert ":ref:`sphx_glr_auto_examples_plot_numpy_matplotlib.py`" in content
    parenthetical = "(``plot_numpy_matplotlib.py``)"
    assert parenthetical in content
    # HTML output
    timings_html = op.join(out_dir, "auto_examples", "sg_execution_times.html")
    assert op.isfile(timings_html)
    with codecs.open(timings_html, "r", "utf-8") as fid:
        content = fid.read()
    assert 'href="plot_numpy_matplotlib.html' in content
    # printed
    status = sphinx_app._status.getvalue()
    fname = op.join("..", "examples", "plot_numpy_matplotlib.py")
    assert f"- {fname}: " in status


def test_api_usage(sphinx_app):
    """Test that an api usage page is created."""
    out_dir = sphinx_app.outdir
    src_dir = sphinx_app.srcdir
    # the rst file was empty but is removed in post-processing
    api_rst = op.join(src_dir, "sg_api_usage.rst")
    assert not op.isfile(api_rst)
    # HTML output
    api_html = op.join(out_dir, "sg_api_usage.html")
    assert op.isfile(api_html)
    with codecs.open(api_html, "r", "utf-8") as fid:
        content = fid.read()
    has_graphviz = _has_graphviz()
    # spot check references
    assert (
        'href="gen_modules/sphinx_gallery.gen_gallery.html'
        '#sphinx_gallery.gen_gallery.setup"'
    ) in content
    # check used and unused
    if has_graphviz:
        assert 'alt="API unused entries graph"' in content
        if sphinx_app.config.sphinx_gallery_conf["show_api_usage"]:
            assert 'alt="sphinx_gallery usage graph"' in content
        else:
            assert 'alt="sphinx_gallery usage graph"' not in content
        # check graph output
        assert 'src="_images/graphviz-' in content
    else:
        assert 'alt="API unused entries graph"' not in content
        assert 'alt="sphinx_gallery usage graph"' not in content
    # printed
    status = sphinx_app._status.getvalue()
    fname = op.join("..", "examples", "plot_numpy_matplotlib.py")
    assert f"- {fname}: " in status


def test_optipng(sphinx_app):
    """Test that optipng is detected."""
    status = sphinx_app._status.getvalue()
    w = sphinx_app._warning.getvalue()
    substr = "will not be optimized"
    if _has_optipng():
        assert substr not in w
    else:
        assert substr in w
    assert "optipng version" not in status.lower()  # catch the --version


def test_junit(sphinx_app, tmpdir):
    out_dir = sphinx_app.outdir
    junit_file = op.join(out_dir, "sphinx-gallery", "junit-results.xml")
    assert op.isfile(junit_file)
    with codecs.open(junit_file, "r", "utf-8") as fid:
        contents = fid.read()
    assert contents.startswith("<?xml")
    assert 'errors="0" failures="0"' in contents
    assert f'tests="{N_EXAMPLES}"' in contents
    assert "local_module" not in contents  # it's not actually run as an ex
    assert "expected example failure" in contents
    assert "<failure message" not in contents
    src_dir = sphinx_app.srcdir
    new_src_dir = op.join(str(tmpdir), "src")
    shutil.copytree(op.join(src_dir, "../"), new_src_dir)
    del src_dir
    new_src_dir = op.join(new_src_dir, "doc")
    new_out_dir = op.join(new_src_dir, "_build", "html")
    new_toctree_dir = op.join(new_src_dir, "_build", "toctrees")
    passing_fname = op.join(new_src_dir, "../examples", "plot_numpy_matplotlib.py")
    failing_fname = op.join(
        new_src_dir, "../examples", "future", "plot_future_imports_broken.py"
    )
    print("Names", passing_fname, failing_fname)
    shutil.move(passing_fname, passing_fname + ".temp")
    shutil.move(failing_fname, passing_fname)
    shutil.move(passing_fname + ".temp", failing_fname)
    with docutils_namespace():
        app = Sphinx(
            new_src_dir,
            new_src_dir,
            new_out_dir,
            new_toctree_dir,
            buildername="html",
            status=StringIO(),
        )
        # need to build within the context manager
        # for automodule and backrefs to work
        with pytest.raises(ExtensionError, match="Here is a summary of the "):
            app.build(False, [])
    junit_file = op.join(new_out_dir, "sphinx-gallery", "junit-results.xml")
    assert op.isfile(junit_file)
    with codecs.open(junit_file, "r", "utf-8") as fid:
        contents = fid.read()
    assert 'errors="0" failures="2"' in contents
    # this time we only ran the stale files
    assert f'tests="{N_FAILING + 1}"' in contents
    assert '<failure message="RuntimeError: Forcing' in contents
    assert "Passed even though it was marked to fail" in contents


def test_run_sphinx(sphinx_app):
    """Test basic outputs."""
    out_dir = str(sphinx_app.outdir)
    out_files = os.listdir(out_dir)
    assert "index.html" in out_files
    assert "auto_examples" in out_files
    assert "auto_examples_with_rst" in out_files
    assert "auto_examples_rst_index" in out_files
    generated_examples_dir = op.join(out_dir, "auto_examples")
    assert op.isdir(generated_examples_dir)
    # make sure that indices are properly being passed forward...
    files_to_check = [
        "auto_examples_rst_index/examp_subdir1/index.html",
        "auto_examples_rst_index/examp_subdir2/index.html",
        "auto_examples_rst_index/index.html",
    ]
    for f in files_to_check:
        assert op.isfile(out_dir + "/" + f)
    status = sphinx_app._status.getvalue()
    assert f"executed {N_GOOD} out of {N_EXAMPLES}" in status
    assert "after excluding 0" in status
    # intentionally have a bad URL in references
    warning = sphinx_app._warning.getvalue()
    want = ".*fetching .*wrong_url.*404.*"
    assert re.match(want, warning, re.DOTALL) is not None, warning


def test_thumbnail_path(sphinx_app, tmpdir):
    """Test sphinx_gallery_thumbnail_path."""
    import numpy as np

    # Make sure our thumbnail matches what it should be
    fname_orig = op.join(sphinx_app.srcdir, "_static_nonstandard", "demo.png")
    fname_thumb = op.join(
        sphinx_app.outdir, "_images", "sphx_glr_plot_second_future_imports_thumb.png"
    )
    fname_new = str(tmpdir.join("new.png"))
    scale_image(
        fname_orig, fname_new, *sphinx_app.config.sphinx_gallery_conf["thumbnail_size"]
    )
    Image = _get_image()
    orig = np.asarray(Image.open(fname_thumb))
    new = np.asarray(Image.open(fname_new))
    assert new.shape[:2] == orig.shape[:2]
    assert new.shape[2] in (3, 4)  # optipng can strip the alpha channel
    corr = np.corrcoef(new[..., :3].ravel(), orig[..., :3].ravel())[0, 1]
    assert corr > 0.99


def test_negative_thumbnail_config(sphinx_app, tmpdir):
    """Test 'sphinx_gallery_thumbnail_number' config correct for negative numbers."""
    import numpy as np

    # Make sure our thumbnail is the 2nd (last) image
    fname_orig = op.join(
        sphinx_app.outdir, "_images", "sphx_glr_plot_matplotlib_alt_002.png"
    )
    fname_thumb = op.join(
        sphinx_app.outdir, "_images", "sphx_glr_plot_matplotlib_alt_thumb.png"
    )
    fname_new = str(tmpdir.join("new.png"))
    scale_image(
        fname_orig, fname_new, *sphinx_app.config.sphinx_gallery_conf["thumbnail_size"]
    )
    Image = _get_image()
    orig = np.asarray(Image.open(fname_thumb))
    new = np.asarray(Image.open(fname_new))
    assert new.shape[:2] == orig.shape[:2]
    assert new.shape[2] in (3, 4)  # optipng can strip the alpha channel
    corr = np.corrcoef(new[..., :3].ravel(), orig[..., :3].ravel())[0, 1]
    assert corr > 0.99


def test_command_line_args_img(sphinx_app):
    generated_examples_dir = op.join(sphinx_app.outdir, "auto_examples")
    thumb_fname = "../_images/sphx_glr_plot_command_line_args_thumb.png"
    file_fname = op.join(generated_examples_dir, thumb_fname)
    assert op.isfile(file_fname), file_fname


def test_image_formats(sphinx_app):
    """Test Image format support."""
    generated_examples_dir = op.join(sphinx_app.outdir, "auto_examples")
    generated_examples_index = op.join(generated_examples_dir, "index.html")
    with codecs.open(generated_examples_index, "r", "utf-8") as fid:
        html = fid.read()
    thumb_fnames = [
        "../_images/sphx_glr_plot_svg_thumb.svg",
        "../_images/sphx_glr_plot_numpy_matplotlib_thumb.png",
        "../_images/sphx_glr_plot_animation_thumb.gif",
        "../_images/sphx_glr_plot_webp_thumb.webp",
    ]
    for thumb_fname in thumb_fnames:
        file_fname = op.join(generated_examples_dir, thumb_fname)
        assert op.isfile(file_fname), file_fname
        want_html = f'src="{thumb_fname}"'
        assert want_html in html
    # the original GIF does not get copied because it's not used in the
    # reST/HTML, so can't add it to this check
    for ex, ext, nums, extra in (
        ("plot_svg", "svg", [1], None),
        ("plot_numpy_matplotlib", "png", [1], None),
        ("plot_animation", "png", [1, 3], "function Animation"),
        ("plot_webp", "webp", [1], None),
    ):
        html_fname = op.join(generated_examples_dir, f"{ex}.html")
        with codecs.open(html_fname, "r", "utf-8") as fid:
            html = fid.read()
        for num in nums:
            img_fname0 = f"../_images/sphx_glr_{ex}_{num:03}.{ext}"
            file_fname = op.join(generated_examples_dir, img_fname0)
            assert op.isfile(file_fname), file_fname
            want_html = f'src="{img_fname0}"'
            assert want_html in html
            img_fname2 = f"../_images/sphx_glr_{ex}_{num:03}_2_00x.{ext}"
            file_fname2 = op.join(generated_examples_dir, img_fname2)
            want_html = f'srcset="{img_fname0}, {img_fname2} 2.00x"'
            if ext in ("png", "jpg", "svg", "webp"):
                # check 2.00x (tests directive)
                assert op.isfile(file_fname2), file_fname2
                assert want_html in html

        if extra is not None:
            assert extra in html


def test_repr_html_classes(sphinx_app):
    """Test appropriate _repr_html_ classes."""
    example_file = op.join(sphinx_app.outdir, "auto_examples", "plot_repr.html")
    with codecs.open(example_file, "r", "utf-8") as fid:
        lines = fid.read()
    assert 'div class="output_subarea output_html rendered_html output_result"' in lines
    assert "gallery-rendered-html.css" in lines


def test_embed_links_and_styles(sphinx_app):
    """Test that links and styles are embedded properly in doc."""
    out_dir = sphinx_app.outdir
    src_dir = sphinx_app.srcdir
    examples_dir = op.join(out_dir, "auto_examples")
    assert op.isdir(examples_dir)
    example_files = os.listdir(examples_dir)
    assert "plot_numpy_matplotlib.html" in example_files
    example_file = op.join(examples_dir, "plot_numpy_matplotlib.html")
    with codecs.open(example_file, "r", "utf-8") as fid:
        lines = fid.read()
    # ensure we've linked properly
    assert "#module-matplotlib.colors" in lines
    assert "matplotlib.colors.is_color_like" in lines
    assert (
        'class="sphx-glr-backref-module-matplotlib-colors sphx-glr-backref-type-py-function">'
        in lines
    )  # noqa: E501
    assert "#module-numpy" in lines
    assert "numpy.arange.html" in lines
    assert (
        'class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-function">'
        in lines
    )  # noqa: E501
    assert "#module-matplotlib.pyplot" in lines
    assert "pyplot.html" in lines or "pyplot_summary.html" in lines
    assert ".html#matplotlib.figure.Figure.tight_layout" in lines
    assert "matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot" in lines
    assert "matplotlib_configuration_api.html#matplotlib.RcParams" in lines
    assert "stdtypes.html#list" in lines
    assert "warnings.html#warnings.warn" in lines
    assert "itertools.html#itertools.compress" in lines
    assert "numpy.ndarray.html" in lines
    # see issue 617
    id_names = re.search(
        r"sphinx_gallery.backreferences.html#sphinx[_,-]gallery[.,-]backreferences[.,-]identify[_,-]names",  # noqa: E501
        lines,
    )
    assert id_names is not None
    # instances have an extra CSS class
    assert (
        'class="sphx-glr-backref-module-matplotlib-figure sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a>'
        in lines
    )  # noqa: E501
    assert (
        'class="sphx-glr-backref-module-matplotlib-figure sphx-glr-backref-type-py-class"><span class="n">Figure</span></a>'
        in lines
    )  # noqa: E501
    # gh-587: no classes that are only marked as module without type
    assert re.search(r'"sphx-glr-backref-module-\S*"', lines) is None
    assert (
        'class="sphx-glr-backref-module-sphinx_gallery-backreferences sphx-glr-backref-type-py-function"><span class="n">sphinx_gallery</span><span class="o">.</span><span class="n">backreferences</span><span class="o">.</span><span class="n">identify_names</span></a>'
        in lines
    )  # noqa: E501
    # gh-587: np.random.RandomState links properly
    # NumPy has had this linked as numpy.random.RandomState and
    # numpy.random.mtrand.RandomState so we need regex...
    assert (
        re.search(
            r'\.html#numpy\.random\.(mtrand\.?)?RandomState" title="numpy\.random\.(mtrand\.?)?RandomState" class="sphx-glr-backref-module-numpy-random(-mtrand?)? sphx-glr-backref-type-py-class"><span class="n">np</span>',
            lines,
        )
        is not None
    )  # noqa: E501
    assert (
        re.search(
            r'\.html#numpy\.random\.(mtrand\.?)?RandomState" title="numpy\.random\.(mtrand\.?)?RandomState" class="sphx-glr-backref-module-numpy-random(-mtrand?)? sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">rng</span></a>',
            lines,
        )
        is not None
    )  # noqa: E501
    # gh-587: methods of classes in the module currently being documented
    # issue 617 (regex '-'s)
    # instance
    dummy_class_inst = re.search(
        r'sphinx_gallery.backreferences.html#sphinx[_,-]gallery[.,-]backreferences[.,-][D,d]ummy[C,c]lass" title="sphinx_gallery.backreferences.DummyClass" class="sphx-glr-backref-module-sphinx_gallery-backreferences sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">dc</span>',  # noqa: E501
        lines,
    )
    assert dummy_class_inst is not None
    # class
    dummy_class_class = re.search(
        r'sphinx_gallery.backreferences.html#sphinx[_,-]gallery[.,-]backreferences[.,-][D,d]ummy[C,c]lass" title="sphinx_gallery.backreferences.DummyClass" class="sphx-glr-backref-module-sphinx_gallery-backreferences sphx-glr-backref-type-py-class"><span class="n">sphinx_gallery</span><span class="o">.</span><span class="n">backreferences</span><span class="o">.</span><span class="n">DummyClass</span>',  # noqa: E501
        lines,
    )
    assert dummy_class_class is not None
    # method
    dummy_class_meth = re.search(
        r'sphinx_gallery.backreferences.html#sphinx[_,-]gallery[.,-]backreferences[.,-][D,d]ummy[C,c]lass[.,-]run" title="sphinx_gallery.backreferences.DummyClass.run" class="sphx-glr-backref-module-sphinx_gallery-backreferences sphx-glr-backref-type-py-method"><span class="n">dc</span><span class="o">.</span><span class="n">run</span>',  # noqa: E501
        lines,
    )
    assert dummy_class_meth is not None
    # property (Sphinx 2+ calls it a method rather than attribute, so regex)
    dummy_class_prop = re.compile(
        r'sphinx_gallery.backreferences.html#sphinx[_,-]gallery[.,-]backreferences[.,-][D,d]ummy[C,c]lass[.,-]prop" title="sphinx_gallery.backreferences.DummyClass.prop" class="sphx-glr-backref-module-sphinx_gallery-backreferences sphx-glr-backref-type-py-(attribute|method|property)"><span class="n">dc</span><span class="o">.</span><span class="n">prop</span>'
    )  # noqa: E501
    assert dummy_class_prop.search(lines) is not None

    try:
        import memory_profiler  # noqa: F401
    except ImportError:
        assert "memory usage" not in lines
    else:
        assert "memory usage" in lines

    # CSS styles
    assert 'class="sphx-glr-signature"' in lines
    assert 'class="sphx-glr-timing"' in lines
    for kind in ("python", "jupyter"):
        assert (
            f'class="sphx-glr-download sphx-glr-download-{kind} docutils container"'
            in lines
        )  # noqa:E501

    # highlight language
    fname = op.join(src_dir, "auto_examples", "plot_numpy_matplotlib.rst")
    assert op.isfile(fname)
    with codecs.open(fname, "r", "utf-8") as fid:
        rst = fid.read()
    assert ".. code-block:: Python\n" in rst

    # warnings
    want_warn = (
        r".*plot_numpy_matplotlib\.py:[0-9][0-9]: RuntimeWarning: "
        r"This warning should show up in the output.*"
    )
    assert re.match(want_warn, lines, re.DOTALL) is not None
    sys.stdout.write(lines)

    example_file = op.join(examples_dir, "plot_pickle.html")
    with codecs.open(example_file, "r", "utf-8") as fid:
        lines = fid.read()
    assert "joblib.Parallel.html" in lines


def test_backreferences(sphinx_app):
    """Test backreferences."""
    out_dir = sphinx_app.outdir
    mod_file = op.join(out_dir, "gen_modules", "sphinx_gallery.sorting.html")
    with codecs.open(mod_file, "r", "utf-8") as fid:
        lines = fid.read()
    assert "ExplicitOrder" in lines  # in API doc
    assert "plot_second_future_imports.html" in lines  # backref via code use
    assert "FileNameSortKey" in lines  # in API doc
    assert "plot_numpy_matplotlib.html" in lines  # backref via :class: in str
    mod_file = op.join(out_dir, "gen_modules", "sphinx_gallery.backreferences.html")
    with codecs.open(mod_file, "r", "utf-8") as fid:
        lines = fid.read()
    assert "NameFinder" in lines  # in API doc
    assert "plot_future_imports.html" in lines  # backref via doc block
    # rendered file
    html = op.join(out_dir, "auto_examples", "plot_second_future_imports.html")
    assert op.isfile(html)
    with codecs.open(html, "r", "utf-8") as fid:
        html = fid.read()
    assert (
        "sphinx_gallery.sorting.html#sphinx_gallery.sorting.ExplicitOrder" in html
    )  # noqa: E501
    assert (
        "sphinx_gallery.scrapers.html#sphinx_gallery.scrapers.clean_modules" in html
    )  # noqa: E501
    assert "figure_rst.html" not in html  # excluded


@pytest.mark.parametrize(
    "rst_file, example_used_in",
    [
        pytest.param(
            "sphinx_gallery.backreferences.identify_names.examples",
            "plot_numpy_matplotlib",
            id="identify_names",
        ),
        pytest.param(
            "sphinx_gallery.sorting.ExplicitOrder.examples",
            "plot_second_future_imports",
            id="ExplicitOrder",
        ),
    ],
)
def test_backreferences_examples_rst(sphinx_app, rst_file, example_used_in):
    """Test linking to mini-galleries using backreferences_dir."""
    backref_dir = sphinx_app.srcdir
    examples_rst = op.join(backref_dir, "gen_modules", "backreferences", rst_file)
    with codecs.open(examples_rst, "r", "utf-8") as fid:
        lines = fid.read()
    assert example_used_in in lines
    # check the .. raw:: html div count
    n_open = lines.count("<div")
    n_close = lines.count("</div")
    assert n_open == n_close


def test_backreferences_examples_html(sphinx_app):
    """Test linking to mini-galleries using backreferences_dir."""
    backref_file = op.join(
        sphinx_app.outdir, "gen_modules", "sphinx_gallery.backreferences.html"
    )
    with codecs.open(backref_file, "r", "utf-8") as fid:
        lines = fid.read()
    # Class properties not properly checked on older Sphinx (e.g. 3)
    # so let's use the "id" instead
    regex = re.compile(r'<dt[ \S]*id="sphinx_gallery.backreferences.[ \S]*>')
    n_documented = len(regex.findall(lines))
    possible = "\n".join(line for line in lines.split("\n") if "<dt " in line)
    # identify_names, DummyClass, DummyClass.prop, DummyClass.run, NameFinder
    # NameFinder.get_mapping, NameFinder.visit_Attribute, NameFinder.visit_Import
    # NameFinder.visit_ImportFrom, NameFinder.visit_Name
    assert n_documented == 10, possible
    # identify_names, DummyClass, NameFinder (3); once doc, once left bar (x2)
    n_mini = lines.count("Examples using ")
    assert n_mini == 6
    # only 3 actual mini-gallery divs
    n_div = lines.count('<div class="sphx-glr-thumbnails')
    assert n_div == 3
    # 3 documented uses
    n_thumb = lines.count('<div class="sphx-glr-thumbcontainer')
    assert n_thumb == 3
    # matched opening/closing divs
    n_open = lines.count("<div")
    n_close = lines.count("</div")
    assert n_open == n_close  # should always be equal


def test_logging_std_nested(sphinx_app):
    """Test that nested stdout/stderr uses within a given script work."""
    log_rst = op.join(sphinx_app.srcdir, "auto_examples", "plot_log.rst")
    with codecs.open(log_rst, "r", "utf-8") as fid:
        lines = fid.read()
    assert ".. code-block:: none\n\n    is in the same cell" in lines
    assert ".. code-block:: none\n\n    is not in the same cell" in lines


def _assert_mtimes(list_orig, list_new, different=(), ignore=()):
    """Assert that the correct set of files were changed based on mtime."""
    import numpy as np
    from numpy.testing import assert_allclose

    assert [op.basename(x) for x in list_orig] == [op.basename(x) for x in list_new]
    # This is probably not totally specific/correct, but this fails on 4.0.0
    # and not on other builds (e.g., 4.5) so hopefully good enough until we
    # drop 4.x support
    good_sphinx = Version(sphinx_version) >= Version("4.1")
    for orig, new in zip(list_orig, list_new):
        check_name = op.splitext(op.basename(orig))[0]
        if check_name.endswith("_codeobj"):
            check_name = check_name[:-8]
        if check_name in different:
            if good_sphinx:
                assert np.abs(op.getmtime(orig) - op.getmtime(new)) > 0.1
        elif check_name not in ignore:
            assert_allclose(
                op.getmtime(orig),
                op.getmtime(new),
                atol=1e-3,
                rtol=1e-20,
                err_msg=op.basename(orig),
            )


def test_rebuild(tmpdir_factory, sphinx_app):
    """Test examples that haven't been changed aren't run twice."""
    # First run completes in the fixture.
    status = sphinx_app._status.getvalue()
    lines = [line for line in status.split("\n") if "removed" in line]
    want = f".*{N_RST} added, 0 changed, 0 removed.*"
    assert re.match(want, status, re.MULTILINE | re.DOTALL) is not None, lines
    want = ".*targets for [2-3] source files that are out of date$.*"
    lines = [line for line in status.split("\n") if "out of date" in line]
    assert re.match(want, status, re.MULTILINE | re.DOTALL) is not None, lines
    lines = [line for line in status.split("\n") if "on MD5" in line]
    want = ".*executed %d out of %d.*after excluding 0 files.*based on MD5.*" % (
        N_GOOD,
        N_EXAMPLES,
    )
    assert re.match(want, status, re.MULTILINE | re.DOTALL) is not None, lines
    old_src_dir = (tmpdir_factory.getbasetemp() / "root_old").strpath
    shutil.copytree(sphinx_app.srcdir, old_src_dir)
    generated_modules_0 = sorted(
        op.join(old_src_dir, "gen_modules", f)
        for f in os.listdir(op.join(old_src_dir, "gen_modules"))
        if op.isfile(op.join(old_src_dir, "gen_modules", f))
    )
    generated_backrefs_0 = sorted(
        op.join(old_src_dir, "gen_modules", "backreferences", f)
        for f in os.listdir(op.join(old_src_dir, "gen_modules", "backreferences"))
    )
    generated_rst_0 = sorted(
        op.join(old_src_dir, "auto_examples", f)
        for f in os.listdir(op.join(old_src_dir, "auto_examples"))
        if f.endswith(".rst")
    )
    generated_pickle_0 = sorted(
        op.join(old_src_dir, "auto_examples", f)
        for f in os.listdir(op.join(old_src_dir, "auto_examples"))
        if f.endswith(".pickle")
    )
    copied_py_0 = sorted(
        op.join(old_src_dir, "auto_examples", f)
        for f in os.listdir(op.join(old_src_dir, "auto_examples"))
        if f.endswith(".py")
    )
    copied_ipy_0 = sorted(
        op.join(old_src_dir, "auto_examples", f)
        for f in os.listdir(op.join(old_src_dir, "auto_examples"))
        if f.endswith(".ipynb")
    )
    assert len(generated_modules_0) > 0
    assert len(generated_backrefs_0) > 0
    assert len(generated_rst_0) > 0
    assert len(generated_pickle_0) > 0
    assert len(copied_py_0) > 0
    assert len(copied_ipy_0) > 0
    assert len(sphinx_app.config.sphinx_gallery_conf["stale_examples"]) == 0
    assert op.isfile(
        op.join(sphinx_app.outdir, "_images", "sphx_glr_plot_numpy_matplotlib_001.png")
    )

    #
    # run a second time, no files should be updated
    #

    src_dir = sphinx_app.srcdir
    del sphinx_app  # don't accidentally use it below
    conf_dir = src_dir
    out_dir = op.join(src_dir, "_build", "html")
    toctrees_dir = op.join(src_dir, "_build", "toctrees")
    time.sleep(0.1)
    with docutils_namespace():
        new_app = Sphinx(
            src_dir,
            conf_dir,
            out_dir,
            toctrees_dir,
            buildername="html",
            status=StringIO(),
        )
        new_app.build(False, [])
    status = new_app._status.getvalue()
    lines = [line for line in status.split("\n") if "0 removed" in line]
    # XXX on Windows this can be more
    if sys.platform.startswith("win"):
        assert (
            re.match(
                ".*[0|1] added, ([1-9]|1[0-4]) changed, 0 removed$.*",
                status,
                re.MULTILINE | re.DOTALL,
            )
            is not None
        ), lines
    else:
        assert (
            re.match(
                ".*[0|1] added, ([1-9]|10) changed, 0 removed$.*",
                status,
                re.MULTILINE | re.DOTALL,
            )
            is not None
        ), lines
    want = ".*executed 0 out of %s.*after excluding %s files.*based on MD5.*" % (
        N_FAILING,
        N_GOOD,
    )
    assert re.match(want, status, re.MULTILINE | re.DOTALL) is not None
    n_stale = len(new_app.config.sphinx_gallery_conf["stale_examples"])
    assert n_stale == N_GOOD
    assert op.isfile(
        op.join(new_app.outdir, "_images", "sphx_glr_plot_numpy_matplotlib_001.png")
    )

    generated_modules_1 = sorted(
        op.join(new_app.srcdir, "gen_modules", f)
        for f in os.listdir(op.join(new_app.srcdir, "gen_modules"))
        if op.isfile(op.join(new_app.srcdir, "gen_modules", f))
    )
    generated_backrefs_1 = sorted(
        op.join(new_app.srcdir, "gen_modules", "backreferences", f)
        for f in os.listdir(op.join(new_app.srcdir, "gen_modules", "backreferences"))
    )
    generated_rst_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".rst")
    )
    generated_pickle_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".pickle")
    )
    copied_py_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".py")
    )
    copied_ipy_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".ipynb")
    )

    # mtimes for modules
    _assert_mtimes(generated_modules_0, generated_modules_1)

    # mtimes for backrefs (gh-394)
    _assert_mtimes(generated_backrefs_0, generated_backrefs_1)

    # generated reST files
    ignore = (
        # these two should almost always be different, but in case we
        # get extremely unlucky and have identical run times
        # on the one script that gets re-run (because it's a fail)...
        "sg_execution_times",
        "sg_api_usage",
        "plot_future_imports_broken",
        "plot_scraper_broken",
    )
    _assert_mtimes(generated_rst_0, generated_rst_1, ignore=ignore)

    # mtimes for pickles
    _assert_mtimes(generated_pickle_0, generated_pickle_1)

    # mtimes for .py files (gh-395)
    _assert_mtimes(copied_py_0, copied_py_1)

    # mtimes for .ipynb files
    _assert_mtimes(copied_ipy_0, copied_ipy_1)

    #
    # run a third and a fourth time, changing one file or running one stale
    #

    for how in ("run_stale", "modify"):
        # modify must be last as this rerun setting tries to run the
        # broken example (subsequent tests depend on it)
        _rerun(
            how,
            src_dir,
            conf_dir,
            out_dir,
            toctrees_dir,
            generated_modules_0,
            generated_backrefs_0,
            generated_rst_0,
            generated_pickle_0,
            copied_py_0,
            copied_ipy_0,
        )


def _rerun(
    how,
    src_dir,
    conf_dir,
    out_dir,
    toctrees_dir,
    generated_modules_0,
    generated_backrefs_0,
    generated_rst_0,
    generated_pickle_0,
    copied_py_0,
    copied_ipy_0,
):
    """Rerun the sphinx build and check that the right files were changed."""
    time.sleep(0.1)
    confoverrides = dict()
    if how == "modify":
        fname = op.join(src_dir, "../examples", "plot_numpy_matplotlib.py")
        with codecs.open(fname, "r", "utf-8") as fid:
            lines = fid.readlines()
        with codecs.open(fname, "w", "utf-8") as fid:
            for line in lines:
                # Make a tiny change that won't affect the recommender
                if "FYI this" in line:
                    line = line.replace("FYI this", "FYA this")
                fid.write(line)
        out_of, excluding = N_FAILING + 1, N_GOOD - 1
        n_stale = N_GOOD - 1
    else:
        assert how == "run_stale"
        confoverrides["sphinx_gallery_conf.run_stale_examples"] = "True"
        confoverrides["sphinx_gallery_conf.filename_pattern"] = "plot_numpy_ma"
        out_of, excluding = 1, 0
        n_stale = 0
    with docutils_namespace():
        new_app = Sphinx(
            src_dir,
            conf_dir,
            out_dir,
            toctrees_dir,
            buildername="html",
            status=StringIO(),
            confoverrides=confoverrides,
        )
        new_app.build(False, [])
    status = new_app._status.getvalue()
    lines = [line for line in status.split("\n") if "source files that" in line]
    lines = "\n".join([how] + lines)
    flags = re.MULTILINE | re.DOTALL
    # for some reason, setting "confoverrides" above causes Sphinx to show
    # all targets out of date, even though they haven't been modified...
    want = f".*targets for {N_RST} source files that are out of date$.*"
    assert re.match(want, status, flags) is not None, lines
    # ... but then later detects that only some have actually changed:
    lines = [line for line in status.split("\n") if "changed," in line]
    # Ones that can change on stale:
    #
    # - auto_examples/future/plot_future_imports_broken
    # - auto_examples/future/sg_execution_times
    # - auto_examples/plot_scraper_broken
    # - auto_examples/sg_execution_times
    # - auto_examples_rst_index/sg_execution_times
    # - auto_examples_with_rst/sg_execution_times
    # - sg_api_usage
    # - sg_execution_times
    #
    # Sometimes it's not all 8, for example when the execution time and
    # memory usage reported ends up being the same.
    #
    # Modifying an example then adds these two:
    # - auto_examples/index
    # - auto_examples/plot_numpy_matplotlib
    if how == "modify":
        n_ch = "([3-9]|10|11)"
    else:
        n_ch = "[1-9]"
    lines = "\n".join([f"\n{how} != {n_ch}:"] + lines)
    want = f".*updating environment:.*[0|1] added, {n_ch} changed, 0 removed.*"
    assert re.match(want, status, flags) is not None, lines
    want = ".*executed 1 out of %s.*after excluding %s files.*based on MD5.*" % (
        out_of,
        excluding,
    )
    assert re.match(want, status, flags) is not None
    got_stale = len(new_app.config.sphinx_gallery_conf["stale_examples"])
    assert got_stale == n_stale
    assert op.isfile(
        op.join(new_app.outdir, "_images", "sphx_glr_plot_numpy_matplotlib_001.png")
    )

    generated_modules_1 = sorted(
        op.join(new_app.srcdir, "gen_modules", f)
        for f in os.listdir(op.join(new_app.srcdir, "gen_modules"))
        if op.isfile(op.join(new_app.srcdir, "gen_modules", f))
    )
    generated_backrefs_1 = sorted(
        op.join(new_app.srcdir, "gen_modules", "backreferences", f)
        for f in os.listdir(op.join(new_app.srcdir, "gen_modules", "backreferences"))
    )
    generated_rst_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".rst")
    )
    generated_pickle_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".pickle")
    )
    copied_py_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".py")
    )
    copied_ipy_1 = sorted(
        op.join(new_app.srcdir, "auto_examples", f)
        for f in os.listdir(op.join(new_app.srcdir, "auto_examples"))
        if f.endswith(".ipynb")
    )

    # mtimes for modules
    _assert_mtimes(generated_modules_0, generated_modules_1)

    # mtimes for backrefs (gh-394)
    _assert_mtimes(generated_backrefs_0, generated_backrefs_1)

    # generated reST files
    different = ("plot_numpy_matplotlib",)
    ignore = (
        # this one should almost always be different, but in case we
        # get extremely unlucky and have identical run times
        # on the one script above that changes...
        "sg_execution_times",
        "sg_api_usage",
        # this one will not change even though it was retried
        "plot_future_imports_broken",
        "plot_scraper_broken",
    )
    # not reliable on Windows and one Ubuntu run
    bad = sys.platform.startswith("win") or os.getenv("BAD_MTIME", "0") == "1"
    if not bad:
        _assert_mtimes(generated_rst_0, generated_rst_1, different, ignore)

        # mtimes for pickles
        use_different = () if how == "run_stale" else different
        _assert_mtimes(generated_pickle_0, generated_pickle_1, ignore=ignore)

        # mtimes for .py files (gh-395)
        _assert_mtimes(copied_py_0, copied_py_1, different=use_different)

        # mtimes for .ipynb files
        _assert_mtimes(copied_ipy_0, copied_ipy_1, different=use_different)


@pytest.mark.parametrize(
    "name, want",
    [
        pytest.param(
            "future/plot_future_imports_broken",
            ".*RuntimeError.*Forcing this example to fail on Python 3.*",
            id="future",
        ),
        pytest.param(
            "plot_scraper_broken",
            ".*ValueError.*zero-size array to reduction.*",
            id="scraper",
        ),
    ],
)
def test_error_messages(sphinx_app, name, want):
    """Test that informative error messages are added."""
    src_dir = Path(sphinx_app.srcdir)
    rst = (src_dir / "auto_examples" / (name + ".rst")).read_text("utf-8")
    assert re.match(want, rst, re.DOTALL) is not None


@pytest.mark.parametrize(
    "name, want",
    [
        pytest.param(
            "future/plot_future_imports_broken",
            ".*RuntimeError.*Forcing this example to fail on Python 3.*",
            id="future",
        ),
        pytest.param(
            "plot_scraper_broken",
            ".*ValueError.*zero-size array to reduction.*",
            id="scraper",
        ),
    ],
)
def test_error_messages_dirhtml(sphinx_dirhtml_app, name, want):
    """Test that informative error messages are added."""
    src_dir = sphinx_dirhtml_app.srcdir
    example_rst = op.join(src_dir, "auto_examples", name + ".rst")
    with codecs.open(example_rst, "r", "utf-8") as fid:
        rst = fid.read()
    rst = rst.replace("\n", " ")
    assert re.match(want, rst) is not None


def test_alt_text_image(sphinx_app):
    """Test alt text for matplotlib images in html and rst."""
    out_dir = sphinx_app.outdir
    src_dir = sphinx_app.srcdir
    # alt text is fig titles, rst
    example_rst = op.join(src_dir, "auto_examples", "plot_matplotlib_alt.rst")
    with codecs.open(example_rst, "r", "utf-8") as fid:
        rst = fid.read()
    # suptitle and axes titles
    assert ":alt: This is a sup title, subplot 1, subplot 2" in rst
    # multiple titles
    assert ":alt: Left Title, Center Title, Right Title" in rst

    # no fig title - alt text is file name, rst
    example_rst = op.join(src_dir, "auto_examples", "plot_numpy_matplotlib.rst")
    with codecs.open(example_rst, "r", "utf-8") as fid:
        rst = fid.read()
    assert ":alt: plot numpy matplotlib" in rst
    # html
    example_html = op.join(out_dir, "auto_examples", "plot_numpy_matplotlib.html")
    with codecs.open(example_html, "r", "utf-8") as fid:
        html = fid.read()
    assert 'alt="plot numpy matplotlib"' in html


def test_alt_text_thumbnail(sphinx_app):
    """Test alt text for thumbnail in html and rst."""
    out_dir = sphinx_app.outdir
    src_dir = sphinx_app.srcdir
    # check gallery index thumbnail, html
    generated_examples_index = op.join(out_dir, "auto_examples", "index.html")
    with codecs.open(generated_examples_index, "r", "utf-8") as fid:
        html = fid.read()
    assert 'alt=""' in html
    # check backreferences thumbnail, html
    backref_html = op.join(out_dir, "gen_modules", "sphinx_gallery.backreferences.html")
    with codecs.open(backref_html, "r", "utf-8") as fid:
        html = fid.read()
    assert 'alt=""' in html
    # check gallery index thumbnail, rst
    generated_examples_index = op.join(src_dir, "auto_examples", "index.rst")
    with codecs.open(generated_examples_index, "r", "utf-8") as fid:
        rst = fid.read()
    assert ":alt:" in rst


def test_backreference_labels(sphinx_app):
    """Tests that backreference labels work."""
    src_dir = sphinx_app.srcdir
    out_dir = sphinx_app.outdir
    # Test backreference label
    backref_rst = op.join(src_dir, "gen_modules", "sphinx_gallery.backreferences.rst")
    with codecs.open(backref_rst, "r", "utf-8") as fid:
        rst = fid.read()
    label = ".. _sphx_glr_backref_sphinx_gallery.backreferences.identify_names:"  # noqa: E501
    assert label in rst
    # Test html link
    index_html = op.join(out_dir, "index.html")
    with codecs.open(index_html, "r", "utf-8") as fid:
        html = fid.read()
    link = 'href="gen_modules/sphinx_gallery.backreferences.html#sphx-glr-backref-sphinx-gallery-backreferences-identify-names">'  # noqa: E501
    assert link in html


@pytest.mark.parametrize(
    "test, nlines, filenamesortkey",
    [
        # first example, no heading
        ("Test 1-N", 5, False),
        # first example, default heading, default level
        ("Test 1-D-D", 7, False),
        # first example, default heading, custom level
        ("Test 1-D-C", 7, False),
        # first example, custom heading, default level
        ("Test 1-C-D", 8, False),
        # both examples, no heading
        ("Test 2-N", 10, True),
        # both examples, default heading, default level
        ("Test 2-D-D", 13, True),
        # both examples, custom heading, custom level
        ("Test 2-C-C", 14, True),
    ],
)
def test_minigallery_directive(sphinx_app, test, nlines, filenamesortkey):
    """Tests the functionality of the minigallery directive."""
    out_dir = sphinx_app.outdir
    minigallery_html = op.join(out_dir, "minigallery.html")
    with codecs.open(minigallery_html, "r", "utf-8") as fid:
        lines = fid.readlines()

    # Regular expressions for matching
    any_heading = re.compile(r"<h([1-6])>.+<\/h\1>")
    explicitorder_example = re.compile(
        r"(?s)<img .+" r"(plot_second_future_imports).+" r"auto_examples\/\1\.html"
    )
    filenamesortkey_example = re.compile(
        r"(?s)<img .+" r"(plot_numpy_matplotlib).+" r"auto_examples\/\1\.html"
    )
    # Heading strings
    heading_str = {
        "Test 1-N": None,
        "Test 1-D-D": r"<h2>Examples using .+ExplicitOrder.+<\/h2>",
        "Test 1-D-C": r"<h3>Examples using .+ExplicitOrder.+<\/h3>",
        "Test 1-C-D": r"<h2>This is a custom heading.*<\/h2>",
        "Test 2-N": None,
        "Test 2-D-D": r"<h2>Examples using one of multiple objects.*<\/h2>",
        "Test 2-C-C": r"<h1>This is a different custom heading.*<\/h1>",
    }

    for i in range(len(lines)):
        if test in lines[i]:
            text = "".join(lines[i : i + nlines])
            print(f"{test}: {text}")
            # Check headings
            if heading_str[test]:
                heading = re.compile(heading_str[test])
                assert heading.search(text) is not None
            else:
                # Confirm there isn't a heading
                assert any_heading.search(text) is None

            # Check for examples
            assert explicitorder_example.search(text) is not None
            if filenamesortkey:
                assert filenamesortkey_example.search(text) is not None
            else:
                assert filenamesortkey_example.search(text) is None


def test_matplotlib_warning_filter(sphinx_app):
    """Test Matplotlib agg warning is removed."""
    out_dir = sphinx_app.outdir
    example_html = op.join(out_dir, "auto_examples", "plot_matplotlib_alt.html")
    with codecs.open(example_html, "r", "utf-8") as fid:
        html = fid.read()
    warning = (
        "Matplotlib is currently using agg, which is a"
        " non-GUI backend, so cannot show the figure."
    )
    assert warning not in html
    warning = "is non-interactive, and thus cannot be shown"
    assert warning not in html


def test_jupyter_notebook_pandoc(sphinx_app):
    """Test using pypandoc."""
    src_dir = sphinx_app.srcdir
    fname = op.join(src_dir, "auto_examples", "plot_numpy_matplotlib.ipynb")
    with codecs.open(fname, "r", "utf-8") as fid:
        md = fid.read()

    md_sg = r"Use :mod:`sphinx_gallery` to link to other packages, like\n:mod:`numpy`, :mod:`matplotlib.colors`, and :mod:`matplotlib.pyplot`."  # noqa
    md_pandoc = r"Use `sphinx_gallery`{.interpreted-text role=\"mod\"} to link to other\npackages, like `numpy`{.interpreted-text role=\"mod\"},\n`matplotlib.colors`{.interpreted-text role=\"mod\"}, and\n`matplotlib.pyplot`{.interpreted-text role=\"mod\"}."  # noqa

    if any(_has_pypandoc()):
        assert md_pandoc in md
    else:
        assert md_sg in md


def test_md5_hash(sphinx_app):
    """Test MD5 hashing."""
    src_dir = sphinx_app.srcdir
    fname = op.join(src_dir, "auto_examples", "plot_log.py.md5")
    expected_md5 = "0edc2de97f96f3b55f8b4a21994931a8"
    with open(fname) as md5_file:
        actual_md5 = md5_file.read()

    assert actual_md5 == expected_md5


def test_interactive_example_logo_exists(sphinx_app):
    """Test that the binder logo path is correct."""
    root = op.join(sphinx_app.outdir, "auto_examples")
    with codecs.open(op.join(root, "plot_svg.html"), "r", "utf-8") as fid:
        html = fid.read()
    path = re.match(
        r'.*<img alt="Launch binder" src="([^"]*)" width=.*\/>.*', html, re.DOTALL
    )
    assert path is not None
    path = path.groups()[0]
    img_fname = op.abspath(op.join(root, path))
    assert "binder_badge_logo" in img_fname  # can have numbers appended
    assert op.isfile(img_fname)
    assert (
        "https://mybinder.org/v2/gh/sphinx-gallery/sphinx-gallery.github.io/master?urlpath=lab/tree/notebooks/auto_examples/plot_svg.ipynb"
        in html
    )  # noqa: E501

    path = re.match(
        r'.*<img alt="Launch JupyterLite" src="([^"]*)" width=.*\/>.*', html, re.DOTALL
    )
    assert path is not None
    path = path.groups()[0]
    img_fname = op.abspath(op.join(root, path))
    assert "jupyterlite_badge_logo" in img_fname  # can have numbers appended
    assert op.isfile(img_fname)


def test_download_and_interactive_note(sphinx_app):
    """Test text saying go to the end to download code or run the example."""
    root = op.join(sphinx_app.outdir, "auto_examples")
    with codecs.open(op.join(root, "plot_svg.html"), "r", "utf-8") as fid:
        html = fid.read()

    pattern = (
        r"to download the full example.+" r"in your browser via JupyterLite or Binder"
    )
    assert re.search(pattern, html)


def test_defer_figures(sphinx_app):
    """Test the deferring of figures."""
    root = op.join(sphinx_app.outdir, "auto_examples")
    fname = op.join(root, "plot_defer_figures.html")
    with codecs.open(fname, "r", "utf-8") as fid:
        html = fid.read()

    # The example has two code blocks with plotting commands, but the first
    # block has the flag ``sphinx_gallery_defer_figures``.  Thus, there should
    # be only one image, not two, in the output.
    assert "../_images/sphx_glr_plot_defer_figures_001.png" in html
    assert "../_images/sphx_glr_plot_defer_figures_002.png" not in html


def test_no_dummy_image(sphinx_app):
    """Test sphinx_gallery_dummy_images NOT created when executable is True."""
    img1 = op.join(
        sphinx_app.srcdir, "auto_examples", "images", "sphx_glr_plot_repr_001.png"
    )
    img2 = op.join(
        sphinx_app.srcdir, "auto_examples", "images", "sphx_glr_plot_repr_002.png"
    )
    assert not op.isfile(img1)
    assert not op.isfile(img2)


def test_jupyterlite_modifications(sphinx_app):
    src_dir = sphinx_app.srcdir
    jupyterlite_notebook_pattern = op.join(
        src_dir, "jupyterlite_contents", "**", "*.ipynb"
    )
    jupyterlite_notebook_filenames = glob.glob(
        jupyterlite_notebook_pattern, recursive=True
    )

    for notebook_filename in jupyterlite_notebook_filenames:
        with open(notebook_filename) as f:
            notebook_content = json.load(f)

        first_cell = notebook_content["cells"][0]
        assert first_cell["cell_type"] == "markdown"
        assert (
            f"JupyterLite-specific change for {notebook_filename}"
            in first_cell["source"]
        )


def test_cpp_rst(sphinx_app):
    cpp_rst = Path(sphinx_app.srcdir) / "auto_examples" / "parse_this.rst"
    content = cpp_rst.read_text()
    assert content.count(".. code-block:: C++") == 3
    assert content.count(":dedent: 1", 1)
    assert "Download C++ source code" in content
    assert "binder-badge" not in content
    assert "lite-badge" not in content
    assert "Download Jupyter notebook" not in content


def test_matlab_rst(sphinx_app):
    matlab_rst = Path(sphinx_app.srcdir) / "auto_examples" / "isentropic.rst"
    content = matlab_rst.read_text()
    assert content.count(".. code-block:: Matlab", 3)
    assert "isentropic, adiabatic flow example\n==============" in content
    assert "Download Matlab source code" in content


def test_julia_rst(sphinx_app):
    julia_rst = Path(sphinx_app.srcdir) / "auto_examples" / "julia_sample.rst"
    content = julia_rst.read_text()
    assert content.count(".. code-block:: Julia", 3)
    assert "Julia example\n=============" in content
    assert "Download Julia source code" in content


def test_recommend_n_examples(sphinx_app):
    """Test correct thumbnails are displayed for an example."""
    pytest.importorskip("numpy")
    root = op.join(sphinx_app.outdir, "auto_examples")
    fname = op.join(root, "plot_defer_figures.html")
    with codecs.open(fname, "r", "utf-8") as fid:
        html = fid.read()

    count = html.count('<div class="sphx-glr-thumbnail-title">')
    n_examples = sphinx_app.config.sphinx_gallery_conf["recommender"]["n_examples"]

    assert '<p class="rubric">Related examples</p>' in html
    assert count == n_examples
    # Check the same 3 related examples are shown
    assert "sphx-glr-auto-examples-plot-repr-py" in html
    assert "sphx-glr-auto-examples-plot-webp-py" in html
    assert "sphx-glr-auto-examples-plot-matplotlib-backend-py" in html
