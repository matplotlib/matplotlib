"""Tests for tinypages build using sphinx extensions."""

import filecmp
from os.path import join as pjoin, dirname, isdir
import pathlib
from subprocess import Popen, PIPE
import sys
import warnings

import pytest

# Only run the tests if Sphinx is installed.
pytest.importorskip('sphinx')

# Docutils is a dependency of Sphinx so it is safe to
# import after we know Sphinx is available.
from docutils.nodes import caption, figure               # noqa: E402

# Sphinx has some deprecation warnings we don't want to turn into errors.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sphinx.application import Sphinx


#: Directory of sources for testing the Sphinx extension.
SRCDIR = pathlib.Path(__file__).parent / 'sphinxext_sources'


class NodeFilter:
    """Test utility class to filter nodes from a Sphinx doctree.

    This is designed to be used with the walkabout() method of nodes. You
    probably want to use the filter_children() class method.

    Parameters
    ----------
    document : node
        The document node.
    classes : list of classes
        The node classes to filter from the document. If None, all classes will
        be accepted resulting in a flattened list of all nodes.

    """
    def __init__(self, document, classes=None):
        self.document = document
        self.nodes = []
        if classes:
            self.classes = tuple(classes)
        else:
            self.classes = None

    def dispatch_visit(self, obj):
        if not self.classes or isinstance(obj, self.classes):
            self.nodes.append(obj)

    def dispatch_departure(self, obj):
        pass

    @classmethod
    def filter_children(cls, document, parent, classes=None):
        """Filter child nodes from a parent node.

        Parameters
        ----------
        document : node
            The main document node.
        parent : node
            The parent node to work on.
        classes : list of classes
            The node classes to filter.

        Returns
        -------
        children : list
            A list of the nodes which are instances of the given classes or
            their subclasses.

        """
        obj = cls(document, classes=classes)
        parent.walkabout(obj)
        return obj.nodes


def build_test_doc(src_dir, build_dir, builder='html'):
    """Build a test document.

    Parameters
    ----------
    src_dir : pathlib.Path
        The location of the sources.
    build_dir : pathlib.Path
        The build directory to use.
    builder : str
        Which builder to use.

    Returns
    -------
    app : sphinx.application.Sphinx
        The Sphinx application that built the document.

    """
    doctree_dir = build_dir / "doctrees"
    output_dir = build_dir / "html"

    # Avoid some internal Sphinx deprecation warnings being turned into errors.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        app = Sphinx(src_dir, src_dir, output_dir, doctree_dir, builder)
        app.build()
    return app


def test_plot_directive_caption(tmpdir):
    """Test the :caption: option of the plot directive.

    """
    # Build the test document.
    localsrc = SRCDIR / "plot_directive_caption"
    build_dir = pathlib.Path(tmpdir)
    app = build_test_doc(localsrc, build_dir)

    # Get the main document and filter out the figures in it.
    index = app.env.get_doctree('index')
    figures = NodeFilter.filter_children(index, index, [figure])

    # The captions we expect to find.
    expected = [
        None,
        'Caption for inline plot.',
        None,
        'This is a caption in the content.',
        'This is a caption in the options.',
        'The content caption should be used instead.',
    ]

    # N.B., each plot directive generates two figures:
    # one HTML only and one for other builders.
    assert len(figures) == 2 * len(expected), \
        "Wrong number of figures in document."

    # Check the caption nodes are correct.
    for i, figurenode in enumerate(figures):
        n = i // 2
        captions = NodeFilter.filter_children(index, figurenode, [caption])

        if expected[n]:
            assert len(captions) > 0, f"Figure {n+1}: no caption found."
            assert len(captions) < 2, f"Figure {n+1}: too many captions."
            assert captions[0].astext().strip() == expected[n], \
                f"Figure {n+1}: wrong caption"
        else:
            assert len(captions) == 0, f"Figure {n+1}: unexpected caption."


def test_tinypages(tmpdir):
    html_dir = pjoin(str(tmpdir), 'html')
    doctree_dir = pjoin(str(tmpdir), 'doctrees')
    # Build the pages with warnings turned into errors
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html', '-d', doctree_dir,
           pjoin(dirname(__file__), 'tinypages'), html_dir]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    out, err = proc.communicate()
    assert proc.returncode == 0, \
        "sphinx build failed with stdout:\n{}\nstderr:\n{}\n".format(out, err)
    if err:
        pytest.fail("sphinx build emitted the following warnings:\n{}"
                    .format(err))

    assert isdir(html_dir)

    def plot_file(num):
        return pjoin(html_dir, 'some_plots-{0}.png'.format(num))

    range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]
    # Plot 5 is range(6) plot
    assert filecmp.cmp(range_6, plot_file(5))
    # Plot 7 is range(4) plot
    assert filecmp.cmp(range_4, plot_file(7))
    # Plot 11 is range(10) plot
    assert filecmp.cmp(range_10, plot_file(11))
    # Plot 12 uses the old range(10) figure and the new range(6) figure
    assert filecmp.cmp(range_10, plot_file('12_00'))
    assert filecmp.cmp(range_6, plot_file('12_01'))
    # Plot 13 shows close-figs in action
    assert filecmp.cmp(range_4, plot_file(13))
    # Plot 14 has included source
    with open(pjoin(html_dir, 'some_plots.html'), 'rb') as fobj:
        html_contents = fobj.read()
    assert b'# Only a comment' in html_contents
    # check plot defined in external file.
    assert filecmp.cmp(range_4, pjoin(html_dir, 'range4.png'))
    assert filecmp.cmp(range_6, pjoin(html_dir, 'range6.png'))
    # check if figure caption made it into html file
    assert b'This is the caption for plot 15.' in html_contents
