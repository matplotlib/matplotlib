"""
Provides a collection of utilities for comparing (image) results.

"""

import atexit
import functools
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryFile

import numpy as np

import matplotlib
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib import _png, cbook

__all__ = ['compare_float', 'compare_images', 'comparable_formats']


def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting `purpose` before the file's
    extension.
    """
    base, ext = os.path.splitext(fname)
    return '%s-%s%s' % (base, purpose, ext)


@cbook.deprecated("3.0")
def compare_float(expected, actual, relTol=None, absTol=None):
    """
    Fail if the floating point values are not close enough, with
    the given message.

    You can specify a relative tolerance, absolute tolerance, or both.

    """
    if relTol is None and absTol is None:
        raise ValueError("You haven't specified a 'relTol' relative "
                         "tolerance or a 'absTol' absolute tolerance "
                         "function argument. You must specify one.")
    msg = ""

    if absTol is not None:
        absDiff = abs(expected - actual)
        if absTol < absDiff:
            template = ['',
                        'Expected: {expected}',
                        'Actual:   {actual}',
                        'Abs diff: {absDiff}',
                        'Abs tol:  {absTol}']
            msg += '\n  '.join([line.format(**locals()) for line in template])

    if relTol is not None:
        # The relative difference of the two values.  If the expected value is
        # zero, then return the absolute value of the difference.
        relDiff = abs(expected - actual)
        if expected:
            relDiff = relDiff / abs(expected)

        if relTol < relDiff:
            # The relative difference is a ratio, so it's always unit-less.
            template = ['',
                        'Expected: {expected}',
                        'Actual:   {actual}',
                        'Rel diff: {relDiff}',
                        'Rel tol:  {relTol}']
            msg += '\n  '.join([line.format(**locals()) for line in template])

    return msg or None


def get_cache_dir():
    cachedir = matplotlib.get_cachedir()
    if cachedir is None:
        raise RuntimeError('Could not find a suitable configuration directory')
    cache_dir = os.path.join(cachedir, 'test_cache')
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except IOError:
        return None
    if not os.access(cache_dir, os.W_OK):
        return None
    return cache_dir


def get_file_hash(path, block_size=2 ** 20):
    md5 = hashlib.md5()
    with open(path, 'rb') as fd:
        while True:
            data = fd.read(block_size)
            if not data:
                break
            md5.update(data)

    if path.endswith('.pdf'):
        from matplotlib import checkdep_ghostscript
        md5.update(checkdep_ghostscript()[1].encode('utf-8'))
    elif path.endswith('.svg'):
        from matplotlib import checkdep_inkscape
        md5.update(checkdep_inkscape().encode('utf-8'))

    return md5.hexdigest()


def make_external_conversion_command(cmd):
    def convert(old, new):
        cmdline = cmd(old, new)
        pipe = subprocess.Popen(cmdline, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        errcode = pipe.wait()
        if not os.path.exists(new) or errcode:
            msg = "Conversion command failed:\n%s\n" % ' '.join(cmdline)
            if stdout:
                msg += "Standard output:\n%s\n" % stdout
            if stderr:
                msg += "Standard error:\n%s\n" % stderr
            raise IOError(msg)

    return convert


# Modified from https://bugs.python.org/issue25567.
_find_unsafe_bytes = re.compile(br'[^a-zA-Z0-9_@%+=:,./-]').search


def _shlex_quote_bytes(b):
    return (b if _find_unsafe_bytes(b) is None
            else b"'" + b.replace(b"'", b"'\"'\"'") + b"'")


class _ConverterError(Exception):
    pass


class _Converter(object):
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
                raise _ConverterError
            buf.extend(c)
            if buf.endswith(terminator):
                return bytes(buf[:-len(terminator)])


class _GSConverter(_Converter):
    def __call__(self, orig, dest):
        if not self._proc:
            self._stdout = TemporaryFile()
            self._proc = subprocess.Popen(
                [matplotlib.checkdep_ghostscript.executable,
                 "-dNOPAUSE", "-sDEVICE=png16m"],
                # As far as I can see, ghostscript never outputs to stderr.
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                self._read_until(b"\nGS")
            except _ConverterError:
                raise OSError("Failed to start Ghostscript")

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
        err = self._read_until(b"GS")
        stack = self._read_until(b">")
        if stack or not os.path.exists(dest):
            stack_size = int(stack[1:]) if stack else 0
            self._proc.stdin.write(b"pop\n" * stack_size)
            # Using the systemencoding should at least get the filenames right.
            raise ImageComparisonFailure(
                (err + b"GS" + stack + b">")
                .decode(sys.getfilesystemencoding(), "replace"))


class _SVGConverter(_Converter):
    def __call__(self, orig, dest):
        if (not self._proc  # First run.
                or self._proc.poll() is not None):  # Inkscape terminated.
            env = os.environ.copy()
            # If one passes e.g. a png file to Inkscape, it will try to
            # query the user for conversion options via a GUI (even with
            # `--without-gui`).  Unsetting `DISPLAY` prevents this (and causes
            # GTK to crash and Inkscape to terminate, but that'll just be
            # reported as a regular exception below).
            env.pop("DISPLAY", None)  # May already be unset.
            # Do not load any user options.
            env["INKSCAPE_PROFILE_DIR"] = os.devnull
            # Old versions of Inkscape (0.48.3.1, used on Travis as of now)
            # seem to sometimes deadlock when stderr is redirected to a pipe,
            # so we redirect it to a temporary file instead.  This is not
            # necessary anymore as of Inkscape 0.92.1.
            stderr = TemporaryFile()
            self._proc = subprocess.Popen(
                ["inkscape", "--without-gui", "--shell"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, env=env)
            # Slight abuse, but makes shutdown handling easier.
            self._proc.stderr = stderr
            try:
                self._read_until(b"\n>")
            except _ConverterError:
                raise OSError("Failed to start Inkscape in interactive mode")

        # Inkscape uses glib's `g_shell_parse_argv`, which has a consistent
        # behavior across platforms, so we can just use `shlex.quote`.
        orig_b, dest_b = map(_shlex_quote_bytes,
                             map(os.fsencode, [orig, dest]))
        if b"\n" in orig_b or b"\n" in dest_b:
            # Who knows whether the current folder name has a newline, or if
            # our encoding is even ASCII compatible...  Just fall back on the
            # slow solution (Inkscape uses `fgets` so it will always stop at a
            # newline).
            return make_external_conversion_command(lambda old, new: [
                'inkscape', '-z', old, '--export-png', new])(orig, dest)
        self._proc.stdin.write(orig_b + b" --export-png=" + dest_b + b"\n")
        self._proc.stdin.flush()
        try:
            self._read_until(b"\n>")
        except _ConverterError:
            # Inkscape's output is not localized but gtk's is, so the output
            # stream probably has a mixed encoding.  Using the filesystem
            # encoding should at least get the filenames right...
            self._stderr.seek(0)
            raise ImageComparisonFailure(
                self._stderr.read().decode(
                    sys.getfilesystemencoding(), "replace"))


def _update_converter():
    gs, gs_v = matplotlib.checkdep_ghostscript()
    if gs_v is not None:
        converter['pdf'] = converter['eps'] = _GSConverter()
    if matplotlib.checkdep_inkscape() is not None:
        converter['svg'] = _SVGConverter()


#: A dictionary that maps filename extensions to functions which
#: themselves map arguments `old` and `new` (filenames) to a list of strings.
#: The list can then be passed to Popen to convert files with that
#: extension to png format.
converter = {}
_update_converter()


def comparable_formats():
    """
    Returns the list of file formats that compare_images can compare
    on this system.

    """
    return ['png', *converter]


def convert(filename, cache):
    """
    Convert the named file to png; return the name of the created file.

    If *cache* is True, the result of the conversion is cached in
    `matplotlib.get_cachedir() + '/test_cache/'`.  The caching is based on a
    hash of the exact contents of the input file.  There is no limit on the
    size of the cache, so it may need to be manually cleared periodically.
    """
    base, extension = filename.rsplit('.', 1)
    if extension not in converter:
        reason = "Don't know how to convert %s files to png" % extension
        from . import is_called_from_pytest
        if is_called_from_pytest():
            import pytest
            pytest.skip(reason)
        else:
            from nose import SkipTest
            raise SkipTest(reason)
    newname = base + '_' + extension + '.png'
    if not os.path.exists(filename):
        raise IOError("'%s' does not exist" % filename)

    # Only convert the file if the destination doesn't already exist or
    # is out of date.
    if (not os.path.exists(newname) or
            os.stat(newname).st_mtime < os.stat(filename).st_mtime):
        if cache:
            cache_dir = get_cache_dir()
        else:
            cache_dir = None

        if cache_dir is not None:
            hash_value = get_file_hash(filename)
            new_ext = os.path.splitext(newname)[1]
            cached_file = os.path.join(cache_dir, hash_value + new_ext)
            if os.path.exists(cached_file):
                shutil.copyfile(cached_file, newname)
                return newname

        converter[extension](filename, newname)

        if cache_dir is not None:
            shutil.copyfile(newname, cached_file)

    return newname


def crop_to_same(actual_path, actual_image, expected_path, expected_image):
    # clip the images to the same size -- this is useful only when
    # comparing eps to pdf
    if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
        aw, ah, ad = actual_image.shape
        ew, eh, ed = expected_image.shape
        actual_image = actual_image[int(aw / 2 - ew / 2):int(
            aw / 2 + ew / 2), int(ah / 2 - eh / 2):int(ah / 2 + eh / 2)]
    return actual_image, expected_image


def calculate_rms(expectedImage, actualImage):
    "Calculate the per-pixel errors, then compute the root mean square error."
    if expectedImage.shape != actualImage.shape:
        raise ImageComparisonFailure(
            "Image sizes do not match expected size: {} "
            "actual size {}".format(expectedImage.shape, actualImage.shape))
    # Convert to float to avoid overflowing finite integer types.
    return np.sqrt(((expectedImage - actualImage).astype(float) ** 2).mean())


def compare_images(expected, actual, tol, in_decorator=False):
    """
    Compare two "image" files checking differences within a tolerance.

    The two given filenames may point to files which are convertible to
    PNG via the `.converter` dictionary. The underlying RMS is calculated
    with the `.calculate_rms` function.

    Parameters
    ----------
    expected : str
        The filename of the expected image.
    actual :str
        The filename of the actual image.
    tol : float
        The tolerance (a color value difference, where 255 is the
        maximal difference).  The test fails if the average pixel
        difference is greater than this value.
    in_decorator : bool
        If called from image_comparison decorator, this should be
        True. (default=False)

    Examples
    --------
    img1 = "./baseline/plot.png"
    img2 = "./output/plot.png"
    compare_images(img1, img2, 0.001):

    """
    if not os.path.exists(actual):
        raise Exception("Output image %s does not exist." % actual)

    if os.stat(actual).st_size == 0:
        raise Exception("Output image file %s is empty." % actual)

    # Convert the image to png
    extension = expected.split('.')[-1]

    if not os.path.exists(expected):
        raise IOError('Baseline image %r does not exist.' % expected)

    if extension != 'png':
        actual = convert(actual, False)
        expected = convert(expected, True)

    # open the image files and remove the alpha channel (if it exists)
    expectedImage = _png.read_png_int(expected)
    actualImage = _png.read_png_int(actual)
    expectedImage = expectedImage[:, :, :3]
    actualImage = actualImage[:, :, :3]

    actualImage, expectedImage = crop_to_same(
        actual, actualImage, expected, expectedImage)

    diff_image = make_test_filename(actual, 'failed-diff')

    if tol <= 0:
        if np.array_equal(expectedImage, actualImage):
            return None

    # convert to signed integers, so that the images can be subtracted without
    # overflow
    expectedImage = expectedImage.astype(np.int16)
    actualImage = actualImage.astype(np.int16)

    rms = calculate_rms(expectedImage, actualImage)

    if rms <= tol:
        return None

    save_diff_image(expected, actual, diff_image)

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
    '''
    Parameters
    ----------
    expected : str
        File path of expected image.
    actual : str
        File path of actual image.
    output : str
        File path to save difference image to.
    '''
    # Drop alpha channels, similarly to compare_images.
    expectedImage = _png.read_png(expected)[..., :3]
    actualImage = _png.read_png(actual)[..., :3]
    actualImage, expectedImage = crop_to_same(
        actual, actualImage, expected, expectedImage)
    expectedImage = np.array(expectedImage).astype(float)
    actualImage = np.array(actualImage).astype(float)
    if expectedImage.shape != actualImage.shape:
        raise ImageComparisonFailure(
            "Image sizes do not match expected size: {} "
            "actual size {}".format(expectedImage.shape, actualImage.shape))
    absDiffImage = np.abs(expectedImage - actualImage)

    # expand differences in luminance domain
    absDiffImage *= 255 * 10
    save_image_np = np.clip(absDiffImage, 0, 255).astype(np.uint8)
    height, width, depth = save_image_np.shape

    # The PDF renderer doesn't produce an alpha channel, but the
    # matplotlib PNG writer requires one, so expand the array
    if depth == 3:
        with_alpha = np.empty((height, width, 4), dtype=np.uint8)
        with_alpha[:, :, 0:3] = save_image_np
        save_image_np = with_alpha

    # Hard-code the alpha channel to fully solid
    save_image_np[:, :, 3] = 255

    _png.write_png(save_image_np, output)
