import contextlib
from distutils.version import StrictVersion
import functools
import inspect
import os
from pathlib import Path
import shutil
import sys
import unittest
import warnings

import matplotlib as mpl
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import cbook
from matplotlib import ft2font
from matplotlib import pyplot as plt
from matplotlib import ticker
from . import is_called_from_pytest
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure


@contextlib.contextmanager
def _cleanup_cm():
    orig_units_registry = matplotlib.units.registry.copy()
    try:
        with warnings.catch_warnings(), matplotlib.rc_context():
            yield
    finally:
        matplotlib.units.registry.clear()
        matplotlib.units.registry.update(orig_units_registry)
        plt.close("all")


class CleanupTestCase(unittest.TestCase):
    """A wrapper for unittest.TestCase that includes cleanup operations."""
    @classmethod
    def setUpClass(cls):
        cls._cm = _cleanup_cm().__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)


@cbook.deprecated("3.0")
class CleanupTest(object):
    setup_class = classmethod(CleanupTestCase.setUpClass.__func__)
    teardown_class = classmethod(CleanupTestCase.tearDownClass.__func__)

    def test(self):
        self._func()


def cleanup(style=None):
    """
    A decorator to ensure that any global state is reset before
    running a test.

    Parameters
    ----------
    style : str, optional
        The name of the style to apply.
    """

    # If cleanup is used without arguments, `style` will be a callable, and we
    # pass it directly to the wrapper generator.  If cleanup if called with an
    # argument, it is a string naming a style, and the function will be passed
    # as an argument to what we return.  This is a confusing, but somewhat
    # standard, pattern for writing a decorator with optional arguments.

    def make_cleanup(func):
        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def wrapped_callable(*args, **kwargs):
                with _cleanup_cm(), matplotlib.style.context(style):
                    yield from func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapped_callable(*args, **kwargs):
                with _cleanup_cm(), matplotlib.style.context(style):
                    func(*args, **kwargs)

        return wrapped_callable

    if isinstance(style, str):
        return make_cleanup
    else:
        result = make_cleanup(style)
        # Default of mpl_test_settings fixture and image_comparison too.
        style = '_classic_test'
        return result


def check_freetype_version(ver):
    if ver is None:
        return True

    if isinstance(ver, str):
        ver = (ver, ver)
    ver = [StrictVersion(x) for x in ver]
    found = StrictVersion(ft2font.__freetype_version__)

    return ver[0] <= found <= ver[1]


def _checked_on_freetype_version(required_freetype_version):
    import pytest
    reason = ("Mismatched version of freetype. "
              "Test requires '%s', you have '%s'" %
              (required_freetype_version, ft2font.__freetype_version__))
    return pytest.mark.xfail(
        not check_freetype_version(required_freetype_version),
        reason=reason, raises=ImageComparisonFailure, strict=False)


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


def _skip_if_format_is_uncomparable(extension):
    import pytest
    return pytest.mark.skipif(
        extension not in comparable_formats(),
        reason='Cannot compare {} files on this system'.format(extension))


def _mark_skip_if_format_is_uncomparable(extension):
    import pytest
    if isinstance(extension, str):
        name = extension
        marks = []
    elif isinstance(extension, tuple):
        # Extension might be a pytest ParameterSet instead of a plain string.
        # Unfortunately, this type is not exposed, so since it's a namedtuple,
        # check for a tuple instead.
        name, = extension.values
        marks = [*extension.marks]
    else:
        # Extension might be a pytest marker instead of a plain string.
        name, = extension.args
        marks = [extension.mark]
    return pytest.param(name,
                        marks=[*marks, _skip_if_format_is_uncomparable(name)])


class _ImageComparisonBase(object):
    """
    Image comparison base class

    This class provides *just* the comparison-related functionality and avoids
    any code that would be specific to any testing framework.
    """
    def __init__(self, tol, remove_text, savefig_kwargs):
        self.func = self.baseline_dir = self.result_dir = None
        self.tol = tol
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs

    def delayed_init(self, func):
        assert self.func is None, "it looks like same decorator used twice"
        self.func = func
        self.baseline_dir, self.result_dir = _image_directories(func)

    def copy_baseline(self, baseline, extension):
        baseline_path = os.path.join(self.baseline_dir, baseline)
        orig_expected_fname = baseline_path + '.' + extension
        if extension == 'eps' and not os.path.exists(orig_expected_fname):
            orig_expected_fname = baseline_path + '.pdf'
        expected_fname = make_test_filename(
            os.path.join(self.result_dir,
                         os.path.basename(orig_expected_fname)),
            'expected')
        if os.path.exists(orig_expected_fname):
            shutil.copyfile(orig_expected_fname, expected_fname)
        else:
            reason = ("Do not have baseline image {} because this "
                      "file does not exist: {}".format(expected_fname,
                                                       orig_expected_fname))
            raise ImageComparisonFailure(reason)
        return expected_fname

    def compare(self, idx, baseline, extension):
        __tracebackhide__ = True
        fignum = plt.get_fignums()[idx]
        fig = plt.figure(fignum)

        if self.remove_text:
            remove_ticks_and_titles(fig)

        actual_fname = (
            os.path.join(self.result_dir, baseline) + '.' + extension)
        kwargs = self.savefig_kwargs.copy()
        if extension == 'pdf':
            kwargs.setdefault('metadata',
                              {'Creator': None, 'Producer': None,
                               'CreationDate': None})
        fig.savefig(actual_fname, **kwargs)

        expected_fname = self.copy_baseline(baseline, extension)
        _raise_on_image_difference(expected_fname, actual_fname, self.tol)


@cbook.deprecated("3.0")
class ImageComparisonTest(CleanupTest, _ImageComparisonBase):
    """
    Nose-based image comparison class

    This class generates tests for a nose-based testing framework. Ideally,
    this class would not be public, and the only publicly visible API would
    be the :func:`image_comparison` decorator. Unfortunately, there are
    existing downstream users of this class (e.g., pytest-mpl) so it cannot yet
    be removed.
    """
    def __init__(self, baseline_images, extensions, tol,
                 freetype_version, remove_text, savefig_kwargs, style):
        _ImageComparisonBase.__init__(self, tol, remove_text, savefig_kwargs)
        self.baseline_images = baseline_images
        self.extensions = extensions
        self.freetype_version = freetype_version
        self.style = style

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
            # Restore original settings before raising errors.
            self.teardown_class()
            raise

    def teardown(self):
        self.teardown_class()

    def nose_runner(self):
        func = self.compare
        func = _checked_on_freetype_version(self.freetype_version)(func)
        funcs = {extension: _skip_if_format_is_uncomparable(extension)(func)
                 for extension in self.extensions}
        for idx, baseline in enumerate(self.baseline_images):
            for extension in self.extensions:
                yield funcs[extension], idx, baseline, extension

    def __call__(self, func):
        self.delayed_init(func)
        import nose.tools

        @functools.wraps(func)
        @nose.tools.with_setup(self.setup, self.teardown)
        def runner_wrapper():
            yield from self.nose_runner()

        return runner_wrapper


def _pytest_image_comparison(baseline_images, extensions, tol,
                             freetype_version, remove_text, savefig_kwargs,
                             style):
    """
    Decorate function with image comparison for pytest.

    This function creates a decorator that wraps a figure-generating function
    with image comparison code. Pytest can become confused if we change the
    signature of the function, so we indirectly pass anything we need via the
    `mpl_image_comparison_parameters` fixture and extra markers.
    """
    import pytest

    extensions = map(_mark_skip_if_format_is_uncomparable, extensions)

    def decorator(func):
        @functools.wraps(func)
        # Parameter indirection; see docstring above and comment below.
        @pytest.mark.usefixtures('mpl_image_comparison_parameters')
        @pytest.mark.parametrize('extension', extensions)
        @pytest.mark.baseline_images(baseline_images)
        # END Parameter indirection.
        @pytest.mark.style(style)
        @_checked_on_freetype_version(freetype_version)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            __tracebackhide__ = True
            img = _ImageComparisonBase(tol=tol, remove_text=remove_text,
                                       savefig_kwargs=savefig_kwargs)
            img.delayed_init(func)
            matplotlib.testing.set_font_settings_for_testing()
            func(*args, **kwargs)

            # Parameter indirection:
            # This is hacked on via the mpl_image_comparison_parameters fixture
            # so that we don't need to modify the function's real signature for
            # any parametrization. Modifying the signature is very very tricky
            # and likely to confuse pytest.
            baseline_images, extension = func.parameters

            assert len(plt.get_fignums()) == len(baseline_images), (
                "Test generated {} images but there are {} baseline images"
                .format(len(plt.get_fignums()), len(baseline_images)))
            for idx, baseline in enumerate(baseline_images):
                img.compare(idx, baseline, extension)

        return wrapper

    return decorator


def image_comparison(baseline_images, extensions=None, tol=0,
                     freetype_version=None, remove_text=False,
                     savefig_kwarg=None,
                     # Default of mpl_test_settings fixture and cleanup too.
                     style='_classic_test'):
    """
    Compare images generated by the test with those specified in
    *baseline_images*, which must correspond, else an `ImageComparisonFailure`
    exception will be raised.

    Parameters
    ----------
    baseline_images : list or None
        A list of strings specifying the names of the images generated by
        calls to :meth:`matplotlib.figure.savefig`.

        If *None*, the test function must use the ``baseline_images`` fixture,
        either as a parameter or with `pytest.mark.usefixtures`. This value is
        only allowed when using pytest.

    extensions : None or list of str
        The list of extensions to test, e.g. ``['png', 'pdf']``.

        If *None*, defaults to all supported extensions: png, pdf, and svg.

        In order to keep the size of the test suite from ballooning, we only
        include the ``svg`` or ``pdf`` outputs if the test is explicitly
        exercising a feature dependent on that backend (see also the
        `check_figures_equal` decorator for that purpose).

    tol : float, optional, default: 0
        The RMS threshold above which the test is considered failed.

    freetype_version : str or tuple
        The expected freetype version or range of versions for this test to
        pass.

    remove_text : bool
        Remove the title and tick text from the figure before comparison.  This
        is useful to make the baseline images independent of variations in text
        rendering between different versions of FreeType.

        This does not remove other, more deliberate, text, such as legends and
        annotations.

    savefig_kwarg : dict
        Optional arguments that are passed to the savefig method.

    style : string
        Optional name for the base style to apply to the image test. The test
        itself can also apply additional styles if desired. Defaults to the
        '_classic_test' style.

    """
    if extensions is None:
        # default extensions to test
        extensions = ['png', 'pdf', 'svg']

    if savefig_kwarg is None:
        #default no kwargs to savefig
        savefig_kwarg = dict()

    if is_called_from_pytest():
        return _pytest_image_comparison(
            baseline_images=baseline_images, extensions=extensions, tol=tol,
            freetype_version=freetype_version, remove_text=remove_text,
            savefig_kwargs=savefig_kwarg, style=style)
    else:
        if baseline_images is None:
            raise ValueError('baseline_images must be specified')

        return ImageComparisonTest(
            baseline_images=baseline_images, extensions=extensions, tol=tol,
            freetype_version=freetype_version, remove_text=remove_text,
            savefig_kwargs=savefig_kwarg, style=style)


def check_figures_equal(*, extensions=("png", "pdf", "svg"), tol=0):
    """
    Decorator for test cases that generate and compare two figures.

    The decorated function must take two arguments, *fig_test* and *fig_ref*,
    and draw the test and reference images on them.  After the function
    returns, the figures are saved and compared.

    This decorator should be preferred over `image_comparison` when possible in
    order to keep the size of the test suite from ballooning.

    Parameters
    ----------
    extensions : list, default: ["png", "pdf", "svg"]
        The extensions to test.
    tol : float
        The RMS threshold above which the test is considered failed.

    Examples
    --------
    Check that calling `Axes.plot` with a single argument plots it against
    ``[0, 1, 2, ...]``::

        @check_figures_equal()
        def test_plot(fig_test, fig_ref):
            fig_test.subplots().plot([1, 3, 5])
            fig_ref.subplots().plot([0, 1, 2], [1, 3, 5])
    """

    def decorator(func):
        import pytest

        _, result_dir = map(Path, _image_directories(func))

        if len(inspect.signature(func).parameters) == 2:
            # Free-standing function.
            @pytest.mark.parametrize("ext", extensions)
            def wrapper(ext):
                fig_test = plt.figure("test")
                fig_ref = plt.figure("reference")
                func(fig_test, fig_ref)
                test_image_path = str(
                    result_dir / (func.__name__ + "." + ext))
                ref_image_path = str(
                    result_dir / (func.__name__ + "-expected." + ext))
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                _raise_on_image_difference(
                    ref_image_path, test_image_path, tol=tol)

        elif len(inspect.signature(func).parameters) == 3:
            # Method.
            @pytest.mark.parametrize("ext", extensions)
            def wrapper(self, ext):
                fig_test = plt.figure("test")
                fig_ref = plt.figure("reference")
                func(self, fig_test, fig_ref)
                test_image_path = str(
                    result_dir / (func.__name__ + "." + ext))
                ref_image_path = str(
                    result_dir / (func.__name__ + "-expected." + ext))
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                _raise_on_image_difference(
                    ref_image_path, test_image_path, tol=tol)

        return wrapper

    return decorator


def _image_directories(func):
    """
    Compute the baseline and result image directories for testing *func*.

    For test module ``foo.bar.test_baz``, the baseline directory is at
    ``foo/bar/baseline_images/test_baz`` and the result directory at
    ``$(pwd)/result_images/test_baz``.  The result directory is created if it
    doesn't exist.
    """
    module_path = Path(sys.modules[func.__module__].__file__)
    baseline_dir = module_path.parent / "baseline_images" / module_path.stem
    result_dir = Path().resolve() / "result_images" / module_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    return str(baseline_dir), str(result_dir)


@cbook.deprecated("3.1", alternative="pytest.mark.backend")
def switch_backend(backend):

    def switch_backend_decorator(func):

        @functools.wraps(func)
        def backend_switcher(*args, **kwargs):
            try:
                prev_backend = mpl.get_backend()
                matplotlib.testing.setup()
                plt.switch_backend(backend)
                return func(*args, **kwargs)
            finally:
                plt.switch_backend(prev_backend)

        return backend_switcher

    return switch_backend_decorator


@cbook.deprecated("3.0")
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
    from subprocess import check_output
    try:
        check_output(cmd)
    except Exception:
        import pytest
        return pytest.mark.skip(reason='missing command: %s' % cmd[0])

    return lambda f: f
