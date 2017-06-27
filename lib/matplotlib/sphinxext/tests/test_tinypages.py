""" Tests for tinypages build using sphinx extensions """

import filecmp
from os.path import join as pjoin, dirname, isdir
import shutil
from subprocess import call, Popen, PIPE
import sys
import tempfile

import pytest

from matplotlib import cbook


HERE = dirname(__file__)
TINY_PAGES = pjoin(HERE, 'tinypages')


def setup_module():
    """Check we have a recent enough version of sphinx installed.
    """
    ret = call([sys.executable, '-msphinx', '--help'],
               stdout=PIPE, stderr=PIPE)
    if ret != 0:
        raise RuntimeError(
            "'{} -msphinx' does not return 0".format(sys.executable))


@cbook.deprecated("2.1", alternative="filecmp.cmp")
def file_same(file1, file2):
    with open(file1, 'rb') as fobj:
        contents1 = fobj.read()
    with open(file2, 'rb') as fobj:
        contents2 = fobj.read()
    return contents1 == contents2


class TestTinyPages(object):
    """Test build and output of tinypages project"""

    @classmethod
    def setup_class(cls):
        cls.page_build = tempfile.mkdtemp()
        try:
            cls.html_dir = pjoin(cls.page_build, 'html')
            cls.doctree_dir = pjoin(cls.page_build, 'doctrees')
            # Build the pages with warnings turned into errors
            cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
                   '-d', cls.doctree_dir,
                   TINY_PAGES,
                   cls.html_dir]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    "'{} -msphinx' failed with stdout:\n{}\nstderr:\n{}\n"
                    .format(sys.executable, out, err))
        except Exception as e:
            shutil.rmtree(cls.page_build)
            raise e

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.page_build)

    def test_some_plots(self):
        assert isdir(self.html_dir)

        def plot_file(num):
            return pjoin(self.html_dir, 'some_plots-{0}.png'.format(num))

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
        with open(pjoin(self.html_dir, 'some_plots.html'), 'rb') as fobj:
            html_contents = fobj.read()
        assert b'# Only a comment' in html_contents
        # check plot defined in external file.
        assert filecmp.cmp(range_4, pjoin(self.html_dir, 'range4.png'))
        assert filecmp.cmp(range_6, pjoin(self.html_dir, 'range6.png'))
        # check if figure caption made it into html file
        assert b'This is the caption for plot 15.' in html_contents
