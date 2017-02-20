from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import functools
import inspect
import os
import sys
import shutil
import warnings
import unittest

# Note - don't import nose up here - import it only as needed in functions. This
# allows other functions here to be used by pytest-based testing suites without
# requiring nose to be installed.


import matplotlib as mpl
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import cbook
from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib import ft2font
from matplotlib.testing.compare import comparable_formats, compare_images, \
     make_test_filename
from . import _copy_metadata, is_called_from_pytest
from .exceptions import ImageComparisonFailure


def _knownfailureif(fail_condition, msg=None, known_exception_class=None):
    """

    Assume a will fail if *fail_condition* is True. *fail_condition*
    may also be False or the string 'indeterminate'.

    *msg* is the error message displayed for the test.

    If *known_exception_class* is not None, the failure is only known
    if the exception is an instance of this class. (Default = None)

    """
    if is_called_from_pytest():
        import pytest
        if fail_condition == 'indeterminate':
            fail_condition, strict = True, False
        else:
            fail_condition, strict = bool(fail_condition), True
        return pytest.mark.xfail(condition=fail_condition, reason=msg,
                                 raises=known_exception_class, strict=strict)
    else:
        from ._nose.decorators import knownfailureif
        return knownfailureif(fail_condition, msg, known_exception_class)


@cbook.deprecated('2.1',
                  alternative='pytest.xfail or import the plugin')
def knownfailureif(fail_condition, msg=None, known_exception_class=None):
    _knownfailureif(fail_condition, msg, known_exception_class)


def _do_cleanup(original_units_registry, original_settings):
    plt.close('all')

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


def cleanup(style=None):
    """
    A decorator to ensure that any global state is reset before
    running a test.

    Parameters
    ----------
    style : str, optional
        The name of the style to apply.
    """

    # If cleanup is used without arguments, `style` will be a
    # callable, and we pass it directly to the wrapper generator.  If
    # cleanup if called with an argument, it is a string naming a
    # style, and the function will be passed as an argument to what we
    # return.  This is a confusing, but somewhat standard, pattern for
    # writing a decorator with optional arguments.

    def make_cleanup(func):
        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def wrapped_callable(*args, **kwargs):
                original_units_registry = matplotlib.units.registry.copy()
                original_settings = mpl.rcParams.copy()
                matplotlib.style.use(style)
                try:
                    for yielded in func(*args, **kwargs):
                        yield yielded
                finally:
                    _do_cleanup(original_units_registry,
                                original_settings)
        else:
            @functools.wraps(func)
            def wrapped_callable(*args, **kwargs):
                original_units_registry = matplotlib.units.registry.copy()
                original_settings = mpl.rcParams.copy()
                matplotlib.style.use(style)
                try:
                    func(*args, **kwargs)
                finally:
                    _do_cleanup(original_units_registry,
                                original_settings)

        return wrapped_callable

    if isinstance(style, six.string_types):
        return make_cleanup
    else:
        result = make_cleanup(style)
        style = 'classic'
        return result


def check_freetype_version(ver):
    if ver is None:
        return True

    from distutils import version
    if isinstance(ver, six.string_types):
        ver = (ver, ver)
    ver = [version.StrictVersion(x) for x in ver]
    found = version.StrictVersion(ft2font.__freetype_version__)

    return found >= ver[0] and found <= ver[1]


def _checked_on_freetype_version(required_freetype_version):
    if check_freetype_version(required_freetype_version):
        return lambda f: f

    reason = ("Mismatched version of freetype. "
              "Test requires '%s', you have '%s'" %
              (required_freetype_version, ft2font.__freetype_version__))
    return _knownfailureif('indeterminate', msg=reason,
                           known_exception_class=ImageComparisonFailure)


def remove_ticks_and_titles(figure):
    figure.suptitle("")
    null_formatter = ticker.NullFormatter()
    for ax in figure.get_axes():
        ax.set_title("")
        ax.xaxis.set_major_formatter(null_formatter)
        ax.xaxis.set_minor_formatter(null_formatter)
        ax.yaxis.set_major_formatter(null_formatter)
        ax.yaxis.set_minor_formatter(null_formatter)
        try:
            ax.zaxis.set_major_formatter(null_formatter)
            ax.zaxis.set_minor_formatter(null_formatter)
        except AttributeError:
            pass


def _raise_on_image_difference(expected, actual, tol):
    __tracebackhide__ = True

    err = compare_images(expected, actual, tol, in_decorator=True)

    if not os.path.exists(expected):
        raise ImageComparisonFailure('image does not exist: %s' % expected)

    if err:
        for key in ["actual", "expected"]:
            err[key] = os.path.relpath(err[key])
        raise ImageComparisonFailure(
            'images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s '
             % err)


def _xfail_if_format_is_uncomparable(extension):
    will_fail = extension not in comparable_formats()
    if will_fail:
        fail_msg = 'Cannot compare %s files on this system' % extension
    else:
        fail_msg = 'No failure expected'

    return _knownfailureif(will_fail, fail_msg,
                           known_exception_class=ImageComparisonFailure)


def _mark_xfail_if_format_is_uncomparable(extension):
    will_fail = extension not in comparable_formats()
    if will_fail:
        fail_msg = 'Cannot compare %s files on this system' % extension
        import pytest
        return pytest.mark.xfail(extension, reason=fail_msg, strict=False,
                                 raises=ImageComparisonFailure)
    else:
        return extension


class ImageComparisonDecorator(CleanupTest):
    def __init__(self, baseline_images, extensions, tol,
                 freetype_version, remove_text, savefig_kwargs, style):
        self.func = self.baseline_dir = self.result_dir = None
        self.baseline_images = baseline_images
        self.extensions = extensions
        self.tol = tol
        self.freetype_version = freetype_version
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs
        self.style = style

    def delayed_init(self, func):
        assert self.func is None, "it looks like same decorator used twice"
        self.func = func
        self.baseline_dir, self.result_dir = _image_directories(func)

    def setup(self):
        func = self.func
        plt.close('all')
        self.setup_class()
        try:
            matplotlib.style.use(self.style)
            matplotlib.testing.set_font_settings_for_testing()
            func()
            assert len(plt.get_fignums()) == len(self.baseline_images), (
                "Test generated {} images but there are {} baseline images"
                .format(len(plt.get_fignums()), len(self.baseline_images)))
        except:
            # Restore original settings before raising errors during the update.
            self.teardown_class()
            raise

    def teardown(self):
        self.teardown_class()

    def copy_baseline(self, baseline, extension):
        baseline_path = os.path.join(self.baseline_dir, baseline)
        orig_expected_fname = baseline_path + '.' + extension
        if extension == 'eps' and not os.path.exists(orig_expected_fname):
            orig_expected_fname = baseline_path + '.pdf'
        expected_fname = make_test_filename(os.path.join(
            self.result_dir, os.path.basename(orig_expected_fname)), 'expected')
        if os.path.exists(orig_expected_fname):
            shutil.copyfile(orig_expected_fname, expected_fname)
        else:
            reason = ("Do not have baseline image {0} because this "
                      "file does not exist: {1}".format(expected_fname,
                                                        orig_expected_fname))
            if is_called_from_pytest():
                import pytest
                pytest.xfail(reason)
            else:
                from ._nose import knownfail
                knownfail(reason)
        return expected_fname

    def compare(self, idx, baseline, extension):
        __tracebackhide__ = True
        fignum = plt.get_fignums()[idx]
        fig = plt.figure(fignum)

        if self.remove_text:
            remove_ticks_and_titles(fig)

        actual_fname = os.path.join(self.result_dir, baseline) + '.' + extension
        kwargs = self.savefig_kwargs.copy()
        if extension == 'pdf':
            kwargs.setdefault('metadata',
                              {'Creator': None, 'Producer': None,
                               'CreationDate': None})
        fig.savefig(actual_fname, **kwargs)

        expected_fname = self.copy_baseline(baseline, extension)
        _raise_on_image_difference(expected_fname, actual_fname, self.tol)

    def nose_runner(self):
        func = self.compare
        func = _checked_on_freetype_version(self.freetype_version)(func)
        funcs = {extension: _xfail_if_format_is_uncomparable(extension)(func)
                 for extension in self.extensions}
        for idx, baseline in enumerate(self.baseline_images):
            for extension in self.extensions:
                yield funcs[extension], idx, baseline, extension

    def pytest_runner(self):
        from pytest import mark

        extensions = map(_mark_xfail_if_format_is_uncomparable,
                         self.extensions)

        if len(set(self.baseline_images)) == len(self.baseline_images):
            @mark.parametrize("extension", extensions)
            @mark.parametrize("idx,baseline", enumerate(self.baseline_images))
            @_checked_on_freetype_version(self.freetype_version)
            def wrapper(idx, baseline, extension):
                __tracebackhide__ = True
                self.compare(idx, baseline, extension)
        else:
            # Some baseline images are repeated, so run this in serial.
            @mark.parametrize("extension", extensions)
            @_checked_on_freetype_version(self.freetype_version)
            def wrapper(extension):
                __tracebackhide__ = True
                for idx, baseline in enumerate(self.baseline_images):
                    self.compare(idx, baseline, extension)


        # sadly we cannot use fixture here because of visibility problems
        # and for for obvious reason avoid `_nose.tools.with_setup`
        wrapper.setup, wrapper.teardown = self.setup, self.teardown

        return wrapper

    def __call__(self, func):
        self.delayed_init(func)
        if is_called_from_pytest():
            return _copy_metadata(func, self.pytest_runner())
        else:
            import nose.tools

            @nose.tools.with_setup(self.setup, self.teardown)
            def runner_wrapper():
                try:
                    for case in self.nose_runner():
                        yield case
                except GeneratorExit:
                    # nose bug...
                    self.teardown()

            return _copy_metadata(func, runner_wrapper)


def image_comparison(baseline_images=None, extensions=None, tol=0,
                     freetype_version=None, remove_text=False,
                     savefig_kwarg=None, style='classic'):
    """
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

      *tol*: (default 0)
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

    return ImageComparisonDecorator(
        baseline_images=baseline_images, extensions=extensions, tol=tol,
        freetype_version=freetype_version, remove_text=remove_text,
        savefig_kwargs=savefig_kwarg, style=style)


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
        if mods.pop(0) != 'tests':
            warnings.warn(("Module '%s' does not live in a parent module "
                "named 'tests'. This is probably ok, but we may not be able "
                "to guess the correct subdirectory containing the baseline "
                "images. If things go wrong please make sure that there is "
                "a parent directory named 'tests' and that it contains a "
                "__init__.py file (can be empty).") % module_name)
        subdir = os.path.join(*mods)

        import imp
        def find_dotted_module(module_name, path=None):
            """A version of imp which can handle dots in the module name.
               As for imp.find_module(), the return value is a 3-element
               tuple (file, pathname, description)."""
            res = None
            for sub_mod in module_name.split('.'):
                try:
                    res = file, path, _ = imp.find_module(sub_mod, path)
                    path = [path]
                    if file is not None:
                        file.close()
                except ImportError:
                    # assume namespace package
                    path = list(sys.modules[sub_mod].__path__)
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
    # Local import to avoid a hard nose dependency and only incur the
    # import time overhead at actual test-time.
    def switch_backend_decorator(func):
        @functools.wraps(func)
        def backend_switcher(*args, **kwargs):
            try:
                prev_backend = mpl.get_backend()
                matplotlib.testing.setup()
                plt.switch_backend(backend)
                result = func(*args, **kwargs)
            finally:
                plt.switch_backend(prev_backend)
            return result

        return _copy_metadata(func, backend_switcher)
    return switch_backend_decorator


def skip_if_command_unavailable(cmd):
    """
    skips a test if a command is unavailable.

    Parameters
    ----------
    cmd : list of str
        must be a complete command which should not
        return a non zero exit code, something like
        ["latex", "-version"]
    """
    from matplotlib.compat.subprocess import check_output
    try:
        check_output(cmd)
    except:
        import pytest
        return pytest.mark.skip(reason='missing command: %s' % cmd[0])

    return lambda f: f
