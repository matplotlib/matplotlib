""" Tests for tinypages build using sphinx extensions """

import shutil
import tempfile

from os.path import (join as pjoin, dirname, isdir)

from subprocess import call, Popen, PIPE

from nose import SkipTest
from nose.tools import assert_true

HERE = dirname(__file__)
TINY_PAGES = pjoin(HERE, 'tinypages')


def setup():
    # Check we have the sphinx-build command
    try:
        ret = call(['sphinx-build', '--help'], stdout=PIPE, stderr=PIPE)
    except OSError:
        raise SkipTest('Need sphinx-build on path for these tests')
    if ret != 0:
        raise RuntimeError('sphinx-build does not return 0')


def file_same(file1, file2):
    with open(file1, 'rb') as fobj:
        contents1 = fobj.read()
    with open(file2, 'rb') as fobj:
        contents2 = fobj.read()
    return contents1 == contents2


class TestTinyPages(object):
    # Test build and output of tinypages project

    @classmethod
    def setup_class(cls):
        cls.page_build = tempfile.mkdtemp()
        try:
            cls.html_dir = pjoin(cls.page_build, 'html')
            cls.doctree_dir = pjoin(cls.page_build, 'doctrees')
            # Build the pages with warnings turned into errors
            cmd = [str('sphinx-build'), '-W', '-b', 'html',
                   '-d', cls.doctree_dir,
                   TINY_PAGES,
                   cls.html_dir]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
        except Exception as e:
            shutil.rmtree(cls.page_build)
            raise e
        if proc.returncode != 0:
            shutil.rmtree(cls.page_build)
            raise RuntimeError('sphinx-build failed with stdout:\n'
                               '{0}\nstderr:\n{1}\n'.format(
                                    out, err))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.page_build)

    def test_some_plots(self):
        assert_true(isdir(self.html_dir))

        def plot_file(num):
            return pjoin(self.html_dir, 'some_plots-{0}.png'.format(num))

        range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]
        # Plot 5 is range(6) plot
        assert_true(file_same(range_6, plot_file(5)))
        # Plot 7 is range(4) plot
        assert_true(file_same(range_4, plot_file(7)))
        # Plot 11 is range(10) plot
        assert_true(file_same(range_10, plot_file(11)))
        # Plot 12 uses the old range(10) figure and the new range(6) figure
        assert_true(file_same(range_10, plot_file('12_00')))
        assert_true(file_same(range_6, plot_file('12_01')))
        # Plot 13 shows close-figs in action
        assert_true(file_same(range_4, plot_file(13)))
        # Plot 14 has included source
        with open(pjoin(self.html_dir, 'some_plots.html'), 'rb') as fobj:
            html_contents = fobj.read()
        assert_true(b'# Only a comment' in html_contents)
