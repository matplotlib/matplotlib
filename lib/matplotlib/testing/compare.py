"""
Utilities for comparing image results.
"""

import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, _image
from matplotlib.testing.exceptions import ImageComparisonFailure

_log = logging.getLogger(__name__)

__all__ = ['calculate_rms', 'comparable_formats', 'compare_images']


def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting *purpose* before the file's extension.
    """
    base, ext = os.path.splitext(fname)
    return f'{base}-{purpose}{ext}'


def _get_cache_path():
    cache_dir = Path(mpl.get_cachedir(), 'test_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_dir():
    return str(_get_cache_path())


def get_file_hash(path, block_size=2 ** 20):
    sha256 = hashlib.sha256(usedforsecurity=False)
    with open(path, 'rb') as fd:
        while True:
            data = fd.read(block_size)
            if not data:
                break
            sha256.update(data)

    if Path(path).suffix == '.pdf':
        sha256.update(str(mpl._get_executable_info("gs").version).encode('utf-8'))
    elif Path(path).suffix == '.svg':
        sha256.update(_resvg_version().encode('utf-8'))

    return sha256.hexdigest()


class _ConverterError(Exception):
    pass


class _Converter:
    def __init__(self):
        self._proc = None
        # Explicitly register deletion from an atexit handler because if we
        # wait until the object is GC'd (which occurs later), then some module
        # globals (e.g. signal.SIGKILL) has already been set to None, and
        # kill() doesn't work anymore...
        atexit.register(self.__del__)

    def __del__(self):
        if self._proc:
            self._proc.kill()
            self._proc.wait()
            for stream in filter(None, [self._proc.stdin,
                                        self._proc.stdout,
                                        self._proc.stderr]):
                stream.close()
            self._proc = None

    def _read_until(self, terminator):
        """Read until the prompt is reached."""
        buf = bytearray()
        while True:
            c = self._proc.stdout.read(1)
            if not c:
                raise _ConverterError(os.fsdecode(bytes(buf)))
            buf.extend(c)
            if buf.endswith(terminator):
                return bytes(buf)


class _MagickConverter:
    def __call__(self, orig, dest):
        try:
            # ImageMagick may not be permitted to follow a symlink, so resolve it
            if orig.is_symlink():
                orig = orig.resolve()
            subprocess.run(
                [mpl._get_executable_info("magick").executable, orig, dest],
                check=True)
        except subprocess.CalledProcessError as e:
            raise _ConverterError() from e


class _GSConverter(_Converter):
    def __call__(self, orig, dest):
        if not self._proc:
            self._proc = subprocess.Popen(
                [mpl._get_executable_info("gs").executable,
                 "-dNOSAFER", "-dNOPAUSE", "-dEPSCrop", "-sDEVICE=png16m"],
                # As far as I can see, ghostscript never outputs to stderr.
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                self._read_until(b"\nGS")
            except _ConverterError as e:
                raise OSError(f"Failed to start Ghostscript:\n\n{e.args[0]}") from None

        def encode_and_escape(name):
            return (os.fsencode(name)
                    .replace(b"\\", b"\\\\")
                    .replace(b"(", br"\(")
                    .replace(b")", br"\)"))

        self._proc.stdin.write(
            b"<< /OutputFile ("
            + encode_and_escape(dest)
            + b") >> setpagedevice ("
            + encode_and_escape(orig)
            + b") run flush\n")
        self._proc.stdin.flush()
        # GS> if nothing left on the stack; GS<n> if n items left on the stack.
        err = self._read_until((b"GS<", b"GS>"))
        stack = self._read_until(b">") if err.endswith(b"GS<") else b""
        if stack or not os.path.exists(dest):
            stack_size = int(stack[:-1]) if stack else 0
            self._proc.stdin.write(b"pop\n" * stack_size)
            # Using the systemencoding should at least get the filenames right.
            raise ImageComparisonFailure(
                (err + stack).decode(sys.getfilesystemencoding(), "replace"))


def _resvg_version():
    from importlib.metadata import version
    return version("resvg-py")


def _convert_svg(orig, dest):
    """
    Render the SVG file *orig* to the PNG file *dest* using resvg.

    Matplotlib's bundled fonts are always made available, and system fonts are
    never used, so that ``svg.fonttype = "none"`` output renders identically
    on all machines.
    """
    import resvg_py

    try:
        png = resvg_py.svg_to_bytes(
            svg_path=os.fspath(orig),
            # Note that the default of dpi=0 fails with "invalid size" on
            # Matplotlib's SVGs (which have sizes in pt).
            dpi=96,
            skip_system_fonts=True,
            font_dirs=[str(cbook._get_data_path("fonts/ttf"))])
    except ValueError as err:
        raise ImageComparisonFailure(str(err)) from err
    Path(dest).write_bytes(png)


def _update_converter():
    try:
        mpl._get_executable_info("magick")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        converter['gif'] = _MagickConverter()
    try:
        mpl._get_executable_info("gs")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        converter['pdf'] = converter['eps'] = _GSConverter()
    try:
        import resvg_py  # noqa: F401
    except ImportError:
        pass
    else:
        converter['svg'] = _convert_svg


#: A dictionary that maps filename extensions to functions which themselves
#: convert between arguments `old` and `new` (filenames).
converter = {}
_update_converter()


def comparable_formats():
    """
    Return the list of file formats that `.compare_images` can compare
    on this system.

    Returns
    -------
    list of str
        E.g. ``['png', 'pdf', 'svg', 'eps']``.

    """
    return ['png', *converter]


def _check_comparable_format(extension):
    if extension not in comparable_formats():
        import pytest
        reason = {
            'gif': 'because ImageMagick is not installed',
            'pdf': 'because Ghostscript is not installed',
            'eps': 'because Ghostscript is not installed',
            'svg': 'because resvg-py is not installed',
        }.get(extension, 'on this system')
        pytest.skip(f"Cannot compare {extension} files {reason}")


def convert(filename, cache):
    """
    Convert the named file to png; return the name of the created file.

    If *cache* is True, the result of the conversion is cached in
    `matplotlib.get_cachedir() + '/test_cache/'`.  The caching is based on a
    hash of the exact contents of the input file.  Old cache entries are
    automatically deleted as needed to keep the size of the cache capped to
    twice the size of all baseline images.
    """
    path = Path(filename)
    if not path.exists():
        raise OSError(f"{path} does not exist")
    _check_comparable_format(path.suffix[1:])
    newpath = path.parent / f"{path.stem}_{path.suffix[1:]}.png"

    # Only convert the file if the destination doesn't already exist or
    # is out of date.
    if not newpath.exists() or newpath.stat().st_mtime < path.stat().st_mtime:
        cache_dir = _get_cache_path() if cache else None

        if cache_dir is not None:
            _register_conversion_cache_cleaner_once()
            hash_value = get_file_hash(path)
            cached_path = cache_dir / (hash_value + newpath.suffix)
            if cached_path.exists():
                _log.debug("For %s: reusing cached conversion.", filename)
                shutil.copyfile(cached_path, newpath)
                return str(newpath)

        _log.debug("For %s: converting to png.", filename)
        convert = converter[path.suffix[1:]]
        convert(path, newpath)

        if cache_dir is not None:
            _log.debug("For %s: caching conversion result.", filename)
            shutil.copyfile(newpath, cached_path)

    return str(newpath)


def _clean_conversion_cache():
    # This will actually ignore mpl_toolkits baseline images, but they're
    # relatively small.
    baseline_images_size = sum(
        path.stat().st_size
        for path in Path(mpl.__file__).parent.glob("**/baseline_images/**/*"))
    # 2x: one full copy of baselines, and one full copy of test results
    # (actually an overestimate: we don't convert png baselines and results).
    max_cache_size = 2 * baseline_images_size
    # Reduce cache until it fits.
    with cbook._lock_path(_get_cache_path()):
        cache_stat = {
            path: path.stat() for path in _get_cache_path().glob("*")}
        cache_size = sum(stat.st_size for stat in cache_stat.values())
        paths_by_atime = sorted(  # Oldest at the end.
            cache_stat, key=lambda path: cache_stat[path].st_atime,
            reverse=True)
        while cache_size > max_cache_size:
            path = paths_by_atime.pop()
            cache_size -= cache_stat[path].st_size
            path.unlink()


@functools.cache  # Ensure this is only registered once.
def _register_conversion_cache_cleaner_once():
    atexit.register(_clean_conversion_cache)


def crop_to_same(actual_path, actual_image, expected_path, expected_image):
    # clip the images to the same size -- this is useful only when
    # comparing eps to pdf
    if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
        aw, ah, ad = actual_image.shape
        ew, eh, ed = expected_image.shape
        actual_image = actual_image[int(aw / 2 - ew / 2):int(
            aw / 2 + ew / 2), int(ah / 2 - eh / 2):int(ah / 2 + eh / 2)]
    return actual_image, expected_image


def calculate_rms(expected_image, actual_image):
    """
    Calculate the per-pixel errors, then compute the root mean square error.
    """
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            f"Image sizes do not match expected size: {expected_image.shape} "
            f"actual size {actual_image.shape}")
    # Convert to float to avoid overflowing finite integer types.
    return np.sqrt(((expected_image - actual_image).astype(float) ** 2).mean())


# NOTE: compare_image and save_diff_image assume that the image does not have
# 16-bit depth, as Pillow converts these to RGB incorrectly.


def _load_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        # If we have anything other than RGB(A) (e.g., a paletted image), convert it
        # to RGBA.
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # In an RGBA image, if the smallest value in the alpha channel is 255, all
        # values in it must be 255, meaning that the image is opaque. If so, discard the
        # alpha channel so that it may compare equal to an RGB image.
        if img.getextrema()[3][0] == 255:
            img = img.convert("RGB")
    return np.asarray(img)


def compare_images(expected, actual, tol, in_decorator=False):
    """
    Compare two "image" files checking differences within a tolerance.

    The two given filenames may point to files which are convertible to
    PNG via the `!converter` dictionary. The underlying RMS is calculated
    in a similar way to the `.calculate_rms` function.

    Parameters
    ----------
    expected : str
        The filename of the expected image.
    actual : str
        The filename of the actual image.
    tol : float
        The tolerance (a color value difference, where 255 is the
        maximal difference).  The test fails if the average pixel
        difference is greater than this value.
    in_decorator : bool
        Determines the output format. If called from image_comparison
        decorator, this should be True. (default=False)

    Returns
    -------
    None or dict or str
        Return *None* if the images are equal within the given tolerance.

        If the images differ, the return value depends on  *in_decorator*.
        If *in_decorator* is true, a dict with the following entries is
        returned:

        - *rms*: The RMS of the image difference.
        - *expected*: The filename of the expected image.
        - *actual*: The filename of the actual image.
        - *diff_image*: The filename of the difference image.
        - *tol*: The comparison tolerance.

        Otherwise, a human-readable multi-line string representation of this
        information is returned.

    Examples
    --------
    ::

        img1 = "./baseline/plot.png"
        img2 = "./output/plot.png"
        compare_images(img1, img2, 0.001)

    """
    actual = os.fspath(actual)
    if not os.path.exists(actual):
        raise Exception(f"Output image {actual} does not exist.")
    if os.stat(actual).st_size == 0:
        raise Exception(f"Output image file {actual} is empty.")

    # Convert the image to png
    expected = os.fspath(expected)
    if not os.path.exists(expected):
        raise OSError(f'Baseline image {expected!r} does not exist.')
    extension = expected.split('.')[-1]
    if extension != 'png':
        actual = convert(actual, cache=True)
        expected = convert(expected, cache=True)

    # open the image files
    expected_image = _load_image(expected)
    actual_image = _load_image(actual)

    actual_image, expected_image = crop_to_same(
        actual, actual_image, expected, expected_image)

    diff_image = make_test_filename(actual, 'failed-diff')

    if tol <= 0:
        if np.array_equal(expected_image, actual_image):
            return None

    rms, abs_diff = _image.calculate_rms_and_diff(expected_image, actual_image)

    if rms <= tol:
        return None

    Image.fromarray(abs_diff).save(diff_image, format="png")

    results = dict(rms=rms, expected=str(expected),
                   actual=str(actual), diff=str(diff_image), tol=tol)

    if not in_decorator:
        # Then the results should be a string suitable for stdout.
        template = ['Error: Image files did not match.',
                    'RMS Value: {rms}',
                    'Expected:  \n    {expected}',
                    'Actual:    \n    {actual}',
                    'Difference:\n    {diff}',
                    'Tolerance: \n    {tol}', ]
        results = '\n  '.join([line.format(**results) for line in template])
    return results


def save_diff_image(expected, actual, output):
    """
    Parameters
    ----------
    expected : str
        File path of expected image.
    actual : str
        File path of actual image.
    output : str
        File path to save difference image to.
    """
    expected_image = _load_image(expected)
    actual_image = _load_image(actual)
    actual_image, expected_image = crop_to_same(
        actual, actual_image, expected, expected_image)
    expected_image = np.array(expected_image, float)
    actual_image = np.array(actual_image, float)
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            f"Image sizes do not match expected size: {expected_image.shape} "
            f"actual size {actual_image.shape}")
    abs_diff = np.abs(expected_image - actual_image)

    # expand differences in luminance domain
    abs_diff *= 10
    abs_diff = np.clip(abs_diff, 0, 255).astype(np.uint8)

    if abs_diff.shape[2] == 4:  # Hard-code the alpha channel to fully solid
        abs_diff[:, :, 3] = 255

    Image.fromarray(abs_diff).save(output, format="png")
