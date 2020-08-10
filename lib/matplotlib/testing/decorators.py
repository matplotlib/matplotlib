import contextlib
import functools
import inspect
import json
import os
from pathlib import Path, PurePosixPath
from platform import uname
import shutil
import string
import subprocess
import sys
import time
import warnings

from packaging.version import parse as parse_version

from matplotlib import _pylab_helpers, cbook, ft2font
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib.style
import matplotlib.testing
import matplotlib.units

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


def _check_freetype_version(ver):
    if ver is None:
        return True

    if isinstance(ver, str):
        ver = (ver, ver)
    ver = [parse_version(x) for x in ver]
    found = parse_version(ft2font.__freetype_version__)

    return ver[0] <= found <= ver[1]


def _checked_on_freetype_version(required_freetype_version):
    import pytest
    return pytest.mark.xfail(
        not _check_freetype_version(required_freetype_version),
        reason=f"Mismatched version of freetype. "
               f"Test requires '{required_freetype_version}', "
               f"you have '{ft2font.__freetype_version__}'",
        raises=ImageComparisonFailure, strict=False)


def remove_ticks_and_titles(figure):
    figure.suptitle("")
    null_formatter = ticker.NullFormatter()
    def remove_ticks(ax):
        """Remove ticks in *ax* and all its child Axes."""
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
        for child in ax.child_axes:
            remove_ticks(child)
    for ax in figure.get_axes():
        remove_ticks(ax)


@contextlib.contextmanager
def _collect_new_figures():
    """
    After::

        with _collect_new_figures() as figs:
            some_code()

    the list *figs* contains the figures that have been created during the
    execution of ``some_code``, sorted by figure number.
    """
    managers = _pylab_helpers.Gcf.figs
    preexisting = [manager for manager in managers.values()]
    new_figs = []
    try:
        yield new_figs
    finally:
        new_managers = sorted([manager for manager in managers.values()
                               if manager not in preexisting],
                              key=lambda manager: manager.num)
        new_figs[:] = [manager.canvas.figure for manager in new_managers]


def _raise_on_image_difference(expected, actual, tol):
    __tracebackhide__ = True

    err = compare_images(expected, actual, tol, in_decorator=True)
    if err:
        for key in ["actual", "expected", "diff"]:
            err[key] = os.path.relpath(err[key])
        raise ImageComparisonFailure(
            ('images not close (RMS %(rms).3f):'
                '\n\t%(actual)s\n\t%(expected)s\n\t%(diff)s') % err)


class _ImageComparisonBase:
    """
    Image comparison base class

    This class provides *just* the comparison-related functionality and avoids
    any code that would be specific to any testing framework.
    """
    def __init__(self, func, tol, remove_text, savefig_kwargs):
        self.func = func
        self.result_dir = _results_directory(func)
        (self.root_dir, mod_dir, image_list, self.md_path) = _baseline_directory(
             func, os.environ.get("MPLTESTIMAGEPATH", None)
         )
        self.image_revs = _load_blame(image_list)
        self.baseline_dir = self.root_dir / mod_dir
        self.tol = tol
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs

    def copy_baseline(self, orig_expected_path):
        expected_fname = Path(make_test_filename(
            self.result_dir / orig_expected_path.name, 'expected'))
        try:
            # os.symlink errors if the target already exists.
            with contextlib.suppress(OSError):
                os.remove(expected_fname)
            try:
                if 'microsoft' in uname().release.lower():
                    raise OSError  # On WSL, symlink breaks silently
                os.symlink(orig_expected_path, expected_fname)
            except OSError:  # On Windows, symlink *may* be unavailable.
                shutil.copyfile(orig_expected_path, expected_fname)
        except OSError as err:
            raise ImageComparisonFailure(
                f"Missing baseline image {expected_fname} because the "
                f"following file cannot be accessed: "
                f"{orig_expected_path}") from err
        return expected_fname

    # TODO add caching?
    def _get_md(self):
        if self.md_path.exists():
            with open(self.md_path) as fin:
                md = {Path(k): v for k, v in json.load(fin).items()}
        else:
            md = {}
            self.md_path.parent.mkdir(parents=True, exist_ok=True)
        return md

    def _write_md(self, md):
        with open(self.md_path, 'w') as fout:
            json.dump(
                {str(PurePosixPath(*k.parts)): v for k, v in md.items()},
                fout,
                sort_keys=True,
                indent='  '
            )

    def _prep_figure(self, fig, baseline, extension):

        if self.remove_text:
            remove_ticks_and_titles(fig)

        actual_path = (self.result_dir / baseline).with_suffix(f'.{extension}')
        kwargs = self.savefig_kwargs.copy()
        if extension == 'pdf':
            kwargs.setdefault('metadata',
                              {'Creator': None, 'Producer': None,
                               'CreationDate': None})
        orig_expected_path = self._compute_baseline_filename(baseline, extension)

        return actual_path, kwargs, orig_expected_path

    def _compute_baseline_filename(self, baseline, extension):
        baseline_path = self.baseline_dir / baseline
        orig_expected_path = baseline_path.with_suffix(f'.{extension}')
        rel_path = orig_expected_path.relative_to(self.root_dir)

        if extension == 'eps' and rel_path not in self.image_revs:
            orig_expected_path = orig_expected_path.with_suffix('.pdf')
            rel_path = orig_expected_path.relative_to(self.root_dir)

        if rel_path not in self.image_revs:
            raise ValueError(f'{rel_path!r} is not known.')
        return orig_expected_path

    def _save_and_close(self, fig, actual_path, kwargs):
        try:
            fig.savefig(actual_path, **kwargs)
        finally:
            # Matplotlib has an autouse fixture to close figures, but this
            # makes things more convenient for third-party users.
            plt.close(fig)

    def generate(self, fig, baseline, extension, *, _lock=False):
        __tracebackhide__ = True
        md = self._get_md()

        actual_path, kwargs, orig_expected_path = self._prep_figure(
            fig, baseline, extension
        )

        lock = (cbook._lock_path(actual_path) if _lock else contextlib.nullcontext())
        with lock:
            self._save_and_close(fig, actual_path, kwargs)

            rel_path = orig_expected_path.relative_to(self.root_dir)
            if rel_path not in self.image_revs and rel_path.suffix == '.eps':
                rel_path = rel_path.with_suffix('.pdf')
            orig_expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual_path, orig_expected_path)

            md[rel_path] = {
                'mpl_version': matplotlib.__version__,
                **{k: self.image_revs[rel_path][k]for k in ('sha', 'rev')}
            }
            self._write_md(md)

    def compare(self, fig, baseline, extension, *, _lock=False):
        __tracebackhide__ = True
        md = self._get_md()
        actual_path, kwargs, orig_expected_path = self._prep_figure(
            fig, baseline, extension
        )

        lock = (cbook._lock_path(actual_path) if _lock else contextlib.nullcontext())
        with lock:
            self._save_and_close(fig, actual_path, kwargs)

            expected_path = self.copy_baseline(orig_expected_path)

            rel_path = actual_path.relative_to(self.result_dir.parent)
            if rel_path not in md and rel_path.suffix == '.eps':
                rel_path = rel_path.with_suffix('.pdf')
            if md[rel_path]['sha'] != self.image_revs[rel_path]['sha']:
                raise RuntimeError("Baseline images do not match checkout.")

            _raise_on_image_difference(expected_path, actual_path, self.tol)


def _pytest_image_comparison(baseline_images, extensions, tol,
                             freetype_version, remove_text, savefig_kwargs,
                             style):
    """
    Decorate function with image comparison for pytest.

    This function creates a decorator that wraps a figure-generating function
    with image comparison code.
    """
    import pytest

    KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY

    def decorator(func):
        old_sig = inspect.signature(func)

        @functools.wraps(func)
        @pytest.mark.parametrize('extension', extensions)
        @matplotlib.style.context(style)
        @_checked_on_freetype_version(freetype_version)
        @functools.wraps(func)
        @pytest.mark.generate_images
        def wrapper(*args, extension, request, **kwargs):
            __tracebackhide__ = True
            if 'extension' in old_sig.parameters:
                kwargs['extension'] = extension
            if 'request' in old_sig.parameters:
                kwargs['request'] = request

            if extension not in comparable_formats():
                reason = {
                    'pdf': 'because Ghostscript is not installed',
                    'eps': 'because Ghostscript is not installed',
                    'svg': 'because Inkscape is not installed',
                }.get(extension, 'on this system')
                pytest.skip(f"Cannot compare {extension} files {reason}")

            img = _ImageComparisonBase(func, tol=tol, remove_text=remove_text,
                                       savefig_kwargs=savefig_kwargs)
            matplotlib.testing.set_font_settings_for_testing()

            with _collect_new_figures() as figs:
                func(*args, **kwargs)

            # If the test is parametrized in any way other than applied via
            # this decorator, then we need to use a lock to prevent two
            # processes from touching the same output file.
            needs_lock = any(
                marker.args[0] != 'extension'
                for marker in request.node.iter_markers('parametrize'))

            if baseline_images is not None:
                our_baseline_images = baseline_images
            else:
                # Allow baseline image list to be produced on the fly based on
                # current parametrization.
                our_baseline_images = request.getfixturevalue(
                    'baseline_images')

            assert len(figs) == len(our_baseline_images), (
                f"Test generated {len(figs)} images but there are "
                f"{len(our_baseline_images)} baseline images")

            generating = request.config.getoption("--generate_images")

            for fig, baseline in zip(figs, our_baseline_images):
                if generating:
                    img.generate(fig, baseline, extension, _lock=needs_lock)
                else:
                    img.compare(fig, baseline, extension, _lock=needs_lock)

        parameters = list(old_sig.parameters.values())
        if 'extension' not in old_sig.parameters:
            parameters += [inspect.Parameter('extension', KEYWORD_ONLY)]
        if 'request' not in old_sig.parameters:
            parameters += [inspect.Parameter("request", KEYWORD_ONLY)]
        new_sig = old_sig.replace(parameters=parameters)
        wrapper.__signature__ = new_sig

        # Reach a bit into pytest internals to hoist the marks from our wrapped
        # function.
        new_marks = getattr(func, 'pytestmark', []) + wrapper.pytestmark
        wrapper.pytestmark = new_marks

        return wrapper

    return decorator


def image_comparison(baseline_images, extensions=None, tol=0,
                     freetype_version=None, remove_text=False,
                     savefig_kwarg=None,
                     # Default of mpl_test_settings fixture and cleanup too.
                     style=("classic", "_classic_test_patch")):
    """
    Compare images generated by the test with those specified in
    *baseline_images*, which must correspond, else an `ImageComparisonFailure`
    exception will be raised.

    Parameters
    ----------
    baseline_images : list or None
        A list of strings specifying the names of the images generated by
        calls to `.Figure.savefig`.

        If *None*, the test function must use the ``baseline_images`` fixture,
        either as a parameter or with `pytest.mark.usefixtures`. This value is
        only allowed when using pytest.

    extensions : None or list of str
        The list of extensions to test, e.g. ``['png', 'pdf']``.

        If *None*, defaults to all supported extensions: png, pdf, and svg.

        When testing a single extension, it can be directly included in the
        names passed to *baseline_images*.  In that case, *extensions* must not
        be set.

        In order to keep the size of the test suite from ballooning, we only
        include the ``svg`` or ``pdf`` outputs if the test is explicitly
        exercising a feature dependent on that backend (see also the
        `check_figures_equal` decorator for that purpose).

    tol : float, default: 0
        The RMS threshold above which the test is considered failed.

        Due to expected small differences in floating-point calculations, on
        32-bit systems an additional 0.06 is added to this threshold.

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

    style : str, dict, or list
        The optional style(s) to apply to the image test. The test itself
        can also apply additional styles if desired. Defaults to ``["classic",
        "_classic_test_patch"]``.
    """

    if baseline_images is not None:
        # List of non-empty filename extensions.
        baseline_exts = [*filter(None, {Path(baseline).suffix[1:]
                                        for baseline in baseline_images})]
        if baseline_exts:
            if extensions is not None:
                raise ValueError(
                    "When including extensions directly in 'baseline_images', "
                    "'extensions' cannot be set as well")
            if len(baseline_exts) > 1:
                raise ValueError(
                    "When including extensions directly in 'baseline_images', "
                    "all baselines must share the same suffix")
            extensions = baseline_exts
            baseline_images = [  # Chop suffix out from baseline_images.
                Path(baseline).stem for baseline in baseline_images]
    if extensions is None:
        # Default extensions to test, if not set via baseline_images.
        extensions = ['png', 'pdf', 'svg']
    if savefig_kwarg is None:
        savefig_kwarg = dict()  # default no kwargs to savefig
    if sys.maxsize <= 2**32:
        tol += 0.06
    return _pytest_image_comparison(
        baseline_images=baseline_images, extensions=extensions, tol=tol,
        freetype_version=freetype_version, remove_text=remove_text,
        savefig_kwargs=savefig_kwarg, style=style)


def check_figures_equal(*, extensions=("png", "pdf", "svg"), tol=0):
    """
    Decorator for test cases that generate and compare two figures.

    The decorated function must take two keyword arguments, *fig_test*
    and *fig_ref*, and draw the test and reference images on them.
    After the function returns, the figures are saved and compared.

    This decorator should be preferred over `image_comparison` when possible in
    order to keep the size of the test suite from ballooning.

    Parameters
    ----------
    extensions : list, default: ["png", "pdf", "svg"]
        The extensions to test.
    tol : float
        The RMS threshold above which the test is considered failed.

    Raises
    ------
    RuntimeError
        If any new figures are created (and not subsequently closed) inside
        the test function.

    Examples
    --------
    Check that calling `.Axes.plot` with a single argument plots it against
    ``[0, 1, 2, ...]``::

        @check_figures_equal()
        def test_plot(fig_test, fig_ref):
            fig_test.subplots().plot([1, 3, 5])
            fig_ref.subplots().plot([0, 1, 2], [1, 3, 5])

    """
    ALLOWED_CHARS = set(string.digits + string.ascii_letters + '_-[]()')
    KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY

    def decorator(func):
        import pytest

        result_dir = _results_directory(func)
        old_sig = inspect.signature(func)

        if not {"fig_test", "fig_ref"}.issubset(old_sig.parameters):
            raise ValueError("The decorated function must have at least the "
                             "parameters 'fig_test' and 'fig_ref', but your "
                             f"function has the signature {old_sig}")

        @pytest.mark.parametrize("ext", extensions)
        def wrapper(*args, ext, request, **kwargs):
            if 'ext' in old_sig.parameters:
                kwargs['ext'] = ext
            if 'request' in old_sig.parameters:
                kwargs['request'] = request

            file_name = "".join(c for c in request.node.name
                                if c in ALLOWED_CHARS)
            try:
                fig_test = plt.figure("test")
                fig_ref = plt.figure("reference")
                with _collect_new_figures() as figs:
                    func(*args, fig_test=fig_test, fig_ref=fig_ref, **kwargs)
                if figs:
                    raise RuntimeError('Number of open figures changed during '
                                       'test. Make sure you are plotting to '
                                       'fig_test or fig_ref, or if this is '
                                       'deliberate explicitly close the '
                                       'new figure(s) inside the test.')
                test_image_path = result_dir / (file_name + "." + ext)
                ref_image_path = result_dir / (file_name + "-expected." + ext)
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                _raise_on_image_difference(
                    ref_image_path, test_image_path, tol=tol
                )
            finally:
                plt.close(fig_test)
                plt.close(fig_ref)

        parameters = [
            param
            for param in old_sig.parameters.values()
            if param.name not in {"fig_test", "fig_ref"}
        ]
        if 'ext' not in old_sig.parameters:
            parameters += [inspect.Parameter("ext", KEYWORD_ONLY)]
        if 'request' not in old_sig.parameters:
            parameters += [inspect.Parameter("request", KEYWORD_ONLY)]
        new_sig = old_sig.replace(parameters=parameters)
        wrapper.__signature__ = new_sig

        # reach a bit into pytest internals to hoist the marks from
        # our wrapped function
        new_marks = getattr(func, "pytestmark", []) + wrapper.pytestmark
        wrapper.pytestmark = new_marks

        return wrapper

    return decorator


# TODO make these functions share a cache
def _load_imagelist(target_file):
    """
    Get the filename, rev, and time stamps from the files.

    This is a sub-set of what _load_blame will provide


    """
    ret = []
    with open(target_file) as fin:
        for ln in fin:
            fname, rev, ts = ln.strip().split(":")
            ret.append([Path(fname), int(rev), float(ts)])

    return {fname: {"rev": rev, "ts": ts} for fname, rev, ts in ret}


def _load_blame(target_file):
    """
    Extract the commit a given test image was last updated in.

    Parameters
    ----------
    target_file : str | Path
       The data file to read.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping of file same to a dictionary mapping to a union of the git
        metadata, the rev number of the file and the timestamp of the last
        revision.

    """
    blame_result = subprocess.run(
        ["git", "blame", "-l", "--line-porcelain", str(target_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    blame_result.check_returncode()
    ret = {}

    cur_line = {}

    for ln in blame_result.stdout.decode().split("\n"):
        if not ln:
            continue

        if ln[0] != "\t":
            if len(cur_line) == 0:
                sha, *_ = ln.split(" ")
                cur_line["sha"] = sha
            else:
                key, _, val = ln.partition(" ")
                cur_line[key] = val
        else:
            fname, rev, ts = ln[1:].strip().split(":")
            cur_line["rev"] = int(rev)
            cur_line["ts"] = float(ts)
            ret[Path(fname)] = cur_line
            cur_line = {}
    return ret


def _write_imagelist(data, *, target_file):
    with open(target_file, "w") as fout:
        for fname, v in sorted(data.items()):
            fout.write(f"{fname}:{v['rev']}:{v['ts']}\n")


def _rev_fname(fname, *, target_file="image_list.txt"):
    data = _load_imagelist()
    old_rev = data[fname]["rev"]
    data[fname] = {"rev": old_rev + 1, "ts": time.time()}
    _write_imagelist(data)


def _baseline_directory(func, external_images):
    """
    Compute the baseline and result image directories for testing *func*.

    For test module ``foo.bar.test_baz``, the baseline directory is at
    ``($base)/foo/bar/baseline_images/test_baz``.

    If *external_images* is not None then it will be used as th ``$base``,
    if not ``$base`` will be the base path of the source code.

    Parameters
    ----------
    func : callable
        The function to compute the file locations for.

    external_images : str or None
        If not None, the root of an external set of baseline images.

    Returns
    -------
    root_dir : Path
        The root of the baseline file tree

    mod_dir : Path
        Path relative to *root_dir* for the images

    image_list : Path
        Path to the list of images (with their versions)

    md_path : Path
        Path to the json meta-data about the generate images

    """
    *pkg_name, mod_name = inspect.getmodule(func).__name__.split('.')
    file_base = Path(inspect.getfile(func)).parent
    if external_images is not None:
        image_base = Path(external_images) / Path(*pkg_name)
    else:
        image_base = file_base
    root_dir = image_base / "baseline_images"
    mod_dir = Path(mod_name)

    image_list = file_base / "image_list.txt"

    return root_dir, mod_dir, image_list, root_dir / 'metadata.json'


def _results_directory(func):
    """
    Compute the result image directories for testing *func*.

    For test module ``foo.bar.test_baz``,  the result directory at
    ``$(pwd)/result_images/test_baz``.

    The result directory is created if it doesn't exist.

    Parameters
    ----------
    func : callable
        The function to compute the file locations for.

    Returns
    -------
    Path
        The path to the directory in which to write the result images
    """
    *pkg_name, mod_name = inspect.getmodule(func).__name__.split('.')
    result_dir = Path().resolve() / "result_images" / mod_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
