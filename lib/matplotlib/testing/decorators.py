from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import functools
import gc
import os
import sys
import shutil
import warnings
import unittest

import nose
import numpy as np

import matplotlib as mpl
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import cbook
from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib import ft2font
from matplotlib.testing.noseclasses import KnownFailureTest, \
     KnownFailureDidNotFailTest, ImageComparisonFailure
from matplotlib.testing.compare import comparable_formats, compare_images, \
     make_test_filename


def knownfailureif(fail_condition, msg=None, known_exception_class=None ):
    """

    Assume a will fail if *fail_condition* is True. *fail_condition*
    may also be False or the string 'indeterminate'.

    *msg* is the error message displayed for the test.

    If *known_exception_class* is not None, the failure is only known
    if the exception is an instance of this class. (Default = None)

    """
    # based on numpy.testing.dec.knownfailureif
    if msg is None:
        msg = 'Test known to fail'
    def known_fail_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        def failer(*args, **kwargs):
            try:
                # Always run the test (to generate images).
                result = f(*args, **kwargs)
            except Exception as err:
                if fail_condition:
                    if known_exception_class is not None:
                        if not isinstance(err,known_exception_class):
                            # This is not the expected exception
                            raise
                    # (Keep the next ultra-long comment so in shows in console.)
                    raise KnownFailureTest(msg) # An error here when running nose means that you don't have the matplotlib.testing.noseclasses:KnownFailure plugin in use.
                else:
                    raise
            if fail_condition and fail_condition != 'indeterminate':
                raise KnownFailureDidNotFailTest(msg)
            return result
        return nose.tools.make_decorator(f)(failer)
    return known_fail_decorator


def _do_cleanup(original_units_registry, original_settings):
    plt.close('all')
    gc.collect()

    mpl.rcParams.clear()
    mpl.rcParams.update(original_settings)
    matplotlib.units.registry.clear()
    matplotlib.units.registry.update(original_units_registry)
    warnings.resetwarnings()  # reset any warning filters set in tests


class CleanupTest(object):
    @classmethod
    def setup_class(cls):
        cls.original_units_registry = matplotlib.units.registry.copy()
        cls.original_settings = mpl.rcParams.copy()
        matplotlib.testing.setup()

    @classmethod
    def teardown_class(cls):
        _do_cleanup(cls.original_units_registry,
                    cls.original_settings)

    def test(self):
        self._func()


class CleanupTestCase(unittest.TestCase):
    '''A wrapper for unittest.TestCase that includes cleanup operations'''
    @classmethod
    def setUpClass(cls):
        import matplotlib.units
        cls.original_units_registry = matplotlib.units.registry.copy()
        cls.original_settings = mpl.rcParams.copy()

    @classmethod
    def tearDownClass(cls):
        _do_cleanup(cls.original_units_registry,
                    cls.original_settings)


def cleanup(func):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        original_units_registry = matplotlib.units.registry.copy()
        original_settings = mpl.rcParams.copy()
        try:
            func(*args, **kwargs)
        finally:
            _do_cleanup(original_units_registry,
                        original_settings)

    return wrapped_function


def check_freetype_version(ver):
    if ver is None:
        return True

    from distutils import version
    if isinstance(ver, six.string_types):
        ver = (ver, ver)
    ver = [version.StrictVersion(x) for x in ver]
    found = version.StrictVersion(ft2font.__freetype_version__)

    return found >= ver[0] and found <= ver[1]

class ImageComparisonTest(CleanupTest):
    @classmethod
    def setup_class(cls):
        cls._initial_settings = mpl.rcParams.copy()
        try:
            matplotlib.style.use(cls._style)
        except:
            # Restore original settings before raising errors during the update.
            mpl.rcParams.clear()
            mpl.rcParams.update(cls._initial_settings)
            raise
        # Because the setup of a CleanupTest might involve
        # modifying a few rcparams, this setup should come
        # last prior to running the image test.
        CleanupTest.setup_class()
        cls.original_settings = cls._initial_settings
        cls._func()

    @classmethod
    def teardown_class(cls):
        CleanupTest.teardown_class()

    @staticmethod
    def remove_text(figure):
        figure.suptitle("")
        for ax in figure.get_axes():
            ax.set_title("")
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            try:
                ax.zaxis.set_major_formatter(ticker.NullFormatter())
                ax.zaxis.set_minor_formatter(ticker.NullFormatter())
            except AttributeError:
                pass

    def test(self):
        baseline_dir, result_dir = _image_directories(self._func)

        for fignum, baseline in zip(plt.get_fignums(), self._baseline_images):
            for extension in self._extensions:
                will_fail = not extension in comparable_formats()
                if will_fail:
                    fail_msg = 'Cannot compare %s files on this system' % extension
                else:
                    fail_msg = 'No failure expected'

                orig_expected_fname = os.path.join(baseline_dir, baseline) + '.' + extension
                if extension == 'eps' and not os.path.exists(orig_expected_fname):
                    orig_expected_fname = os.path.join(baseline_dir, baseline) + '.pdf'
                expected_fname = make_test_filename(os.path.join(
                    result_dir, os.path.basename(orig_expected_fname)), 'expected')
                actual_fname = os.path.join(result_dir, baseline) + '.' + extension
                if os.path.exists(orig_expected_fname):
                    shutil.copyfile(orig_expected_fname, expected_fname)
                else:
                    will_fail = True
                    fail_msg = 'Do not have baseline image %s' % expected_fname

                @knownfailureif(
                    will_fail, fail_msg,
                    known_exception_class=ImageComparisonFailure)
                def do_test():
                    figure = plt.figure(fignum)

                    if self._remove_text:
                        self.remove_text(figure)

                    figure.savefig(actual_fname, **self._savefig_kwarg)

                    err = compare_images(expected_fname, actual_fname,
                                         self._tol, in_decorator=True)

                    try:
                        if not os.path.exists(expected_fname):
                            raise ImageComparisonFailure(
                                'image does not exist: %s' % expected_fname)

                        if err:
                            raise ImageComparisonFailure(
                                'images not close: %(actual)s vs. %(expected)s '
                                '(RMS %(rms).3f)'%err)
                    except ImageComparisonFailure:
                        if not check_freetype_version(self._freetype_version):
                            raise KnownFailureTest(
                                "Mismatched version of freetype.  Test requires '%s', you have '%s'" %
                                (self._freetype_version, ft2font.__freetype_version__))
                        raise

                yield (do_test,)

def image_comparison(baseline_images=None, extensions=None, tol=13,
                     freetype_version=None, remove_text=False,
                     savefig_kwarg=None, style='classic'):
    """
    call signature::

      image_comparison(baseline_images=['my_figure'], extensions=None)

    Compare images generated by the test with those specified in
    *baseline_images*, which must correspond else an
    ImageComparisonFailure exception will be raised.

    Keyword arguments:

      *baseline_images*: list
        A list of strings specifying the names of the images generated
        by calls to :meth:`matplotlib.figure.savefig`.

      *extensions*: [ None | list ]

        If *None*, default to all supported extensions.

        Otherwise, a list of extensions to test. For example ['png','pdf'].

      *tol*: (default 13)
        The RMS threshold above which the test is considered failed.

      *freetype_version*: str or tuple
        The expected freetype version or range of versions for this
        test to pass.

      *remove_text*: bool
        Remove the title and tick text from the figure before
        comparison.  This does not remove other, more deliberate,
        text, such as legends and annotations.

      *savefig_kwarg*: dict
        Optional arguments that are passed to the savefig method.

      *style*: string
        Optional name for the base style to apply to the image
        test. The test itself can also apply additional styles
        if desired. Defaults to the 'classic' style.

    """

    if baseline_images is None:
        raise ValueError('baseline_images must be specified')

    if extensions is None:
        # default extensions to test
        extensions = ['png', 'pdf', 'svg']

    if savefig_kwarg is None:
        #default no kwargs to savefig
        savefig_kwarg = dict()

    def compare_images_decorator(func):
        # We want to run the setup function (the actual test function
        # that generates the figure objects) only once for each type
        # of output file.  The only way to achieve this with nose
        # appears to be to create a test class with "setup_class" and
        # "teardown_class" methods.  Creating a class instance doesn't
        # work, so we use type() to actually create a class and fill
        # it with the appropriate methods.
        name = func.__name__
        # For nose 1.0, we need to rename the test function to
        # something without the word "test", or it will be run as
        # well, outside of the context of our image comparison test
        # generator.
        func = staticmethod(func)
        func.__get__(1).__name__ = str('_private')
        new_class = type(
            name,
            (ImageComparisonTest,),
            {'_func': func,
             '_baseline_images': baseline_images,
             '_extensions': extensions,
             '_tol': tol,
             '_freetype_version': freetype_version,
             '_remove_text': remove_text,
             '_savefig_kwarg': savefig_kwarg,
             '_style': style})

        return new_class
    return compare_images_decorator

def _image_directories(func):
    """
    Compute the baseline and result image directories for testing *func*.
    Create the result directory if it doesn't exist.
    """
    module_name = func.__module__
    if module_name == '__main__':
        # FIXME: this won't work for nested packages in matplotlib.tests
        warnings.warn('test module run as script. guessing baseline image locations')
        script_name = sys.argv[0]
        basedir = os.path.abspath(os.path.dirname(script_name))
        subdir = os.path.splitext(os.path.split(script_name)[1])[0]
    else:
        mods = module_name.split('.')
        if len(mods) >= 3:
            mods.pop(0)
            # mods[0] will be the name of the package being tested (in
            # most cases "matplotlib") However if this is a
            # namespace package pip installed and run via the nose
            # multiprocess plugin or as a specific test this may be
            # missing. See https://github.com/matplotlib/matplotlib/issues/3314
        assert mods.pop(0) == 'tests'
        subdir = os.path.join(*mods)

        import imp
        def find_dotted_module(module_name, path=None):
            """A version of imp which can handle dots in the module name"""
            res = None
            for sub_mod in module_name.split('.'):
                try:
                    res = file, path, _ = imp.find_module(sub_mod, path)
                    path = [path]
                    if file is not None:
                        file.close()
                except ImportError:
                    # assume namespace package
                    path = sys.modules[sub_mod].__path__
                    res = None, path, None
            return res

        mod_file = find_dotted_module(func.__module__)[1]
        basedir = os.path.dirname(mod_file)

    baseline_dir = os.path.join(basedir, 'baseline_images', subdir)
    result_dir = os.path.abspath(os.path.join('result_images', subdir))

    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)

    return baseline_dir, result_dir


def switch_backend(backend):
    def switch_backend_decorator(func):
        def backend_switcher(*args, **kwargs):
            try:
                prev_backend = mpl.get_backend()
                matplotlib.testing.setup()
                plt.switch_backend(backend)
                result = func(*args, **kwargs)
            finally:
                plt.switch_backend(prev_backend)
            return result

        return nose.tools.make_decorator(func)(backend_switcher)
    return switch_backend_decorator
