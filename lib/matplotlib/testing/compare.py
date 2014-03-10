"""
Provides a collection of utilities for comparing (image) results.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import hashlib
import os
import shutil

import numpy as np

import matplotlib
from matplotlib.compat import subprocess
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib import _png
from matplotlib import _get_cachedir
from matplotlib import cbook
from distutils import version

__all__ = ['compare_float', 'compare_images', 'comparable_formats']


def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting `purpose` before the file's
    extension.
    """
    base, ext = os.path.splitext(fname)
    return '%s-%s%s' % (base, purpose, ext)


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
    cachedir = _get_cachedir()
    if cachedir is None:
        raise RuntimeError('Could not find a suitable configuration directory')
    cache_dir = os.path.join(cachedir, 'test_cache')
    if not os.path.exists(cache_dir):
        try:
            cbook.mkdirs(cache_dir)
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
    return md5.hexdigest()


def make_external_conversion_command(cmd):
    def convert(old, new):
        cmdline = cmd(old, new)
        pipe = subprocess.Popen(
            cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def _update_converter():
    gs, gs_v = matplotlib.checkdep_ghostscript()
    if gs_v is not None:
        cmd = lambda old, new: \
            [gs, '-q', '-sDEVICE=png16m', '-dNOPAUSE', '-dBATCH',
             '-sOutputFile=' + new, old]
        converter['pdf'] = make_external_conversion_command(cmd)
        converter['eps'] = make_external_conversion_command(cmd)

    if matplotlib.checkdep_inkscape() is not None:
        cmd = lambda old, new: \
            ['inkscape', '-z', old, '--export-png', new]
        converter['svg'] = make_external_conversion_command(cmd)


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
    return ['png'] + list(six.iterkeys(converter))


def convert(filename, cache):
    """
    Convert the named file into a png file.  Returns the name of the
    created file.

    If *cache* is True, the result of the conversion is cached in
    `matplotlib._get_cachedir() + '/test_cache/'`.  The caching is based
    on a hash of the exact contents of the input file.  The is no limit
    on the size of the cache, so it may need to be manually cleared
    periodically.

    """
    base, extension = filename.rsplit('.', 1)
    if extension not in converter:
        raise ImageComparisonFailure(
            "Don't know how to convert %s files to png" % extension)
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

#: Maps file extensions to a function which takes a filename as its
#: only argument to return a list suitable for execution with Popen.
#: The purpose of this is so that the result file (with the given
#: extension) can be verified with tools such as xmllint for svg.
verifiers = {}

# Turning this off, because it seems to cause multiprocessing issues
if matplotlib.checkdep_xmllint() and False:
    verifiers['svg'] = lambda filename: [
        'xmllint', '--valid', '--nowarning', '--noout', filename]


def verify(filename):
    """Verify the file through some sort of verification tool."""
    if not os.path.exists(filename):
        raise IOError("'%s' does not exist" % filename)
    base, extension = filename.rsplit('.', 1)
    verifier = verifiers.get(extension, None)
    if verifier is not None:
        cmd = verifier(filename)
        pipe = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        errcode = pipe.wait()
        if errcode != 0:
            msg = "File verification command failed:\n%s\n" % ' '.join(cmd)
            if stdout:
                msg += "Standard output:\n%s\n" % stdout
            if stderr:
                msg += "Standard error:\n%s\n" % stderr
            raise IOError(msg)


def crop_to_same(actual_path, actual_image, expected_path, expected_image):
    # clip the images to the same size -- this is useful only when
    # comparing eps to pdf
    if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
        aw, ah = actual_image.shape
        ew, eh = expected_image.shape
        actual_image = actual_image[int(aw / 2 - ew / 2):int(
            aw / 2 + ew / 2), int(ah / 2 - eh / 2):int(ah / 2 + eh / 2)]
    return actual_image, expected_image


def calculate_rms(expectedImage, actualImage):
    "Calculate the per-pixel errors, then compute the root mean square error."
    num_values = np.prod(expectedImage.shape)
    abs_diff_image = abs(expectedImage - actualImage)

    # On Numpy 1.6, we can use bincount with minlength, which is much
    # faster than using histogram
    expected_version = version.LooseVersion("1.6")
    found_version = version.LooseVersion(np.__version__)
    if found_version >= expected_version:
        histogram = np.bincount(abs_diff_image.ravel(), minlength=256)
    else:
        histogram = np.histogram(abs_diff_image, bins=np.arange(257))[0]

    sum_of_squares = np.sum(histogram * np.arange(len(histogram)) ** 2)
    rms = np.sqrt(float(sum_of_squares) / num_values)

    return rms


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

    Example
    -------
    img1 = "./baseline/plot.png"
    img2 = "./output/plot.png"
    compare_images( img1, img2, 0.001 ):

    """
    if not os.path.exists(actual):
        msg = "Output image %s does not exist." % actual
        raise Exception(msg)

    if os.stat(actual).st_size == 0:
        msg = "Output image file %s is empty." % actual
        raise Exception(msg)

    verify(actual)

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

    # convert to signed integers, so that the images can be subtracted without
    # overflow
    expectedImage = expectedImage.astype(np.int16)
    actualImage = actualImage.astype(np.int16)

    rms = calculate_rms(expectedImage, actualImage)

    diff_image = make_test_filename(actual, 'failed-diff')

    if rms <= tol:
        if os.path.exists(diff_image):
            os.unlink(diff_image)
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
    expectedImage = _png.read_png(expected)
    actualImage = _png.read_png(actual)
    actualImage, expectedImage = crop_to_same(
        actual, actualImage, expected, expectedImage)
    expectedImage = np.array(expectedImage).astype(np.float)
    actualImage = np.array(actualImage).astype(np.float)
    assert expectedImage.ndim == actualImage.ndim
    assert expectedImage.shape == actualImage.shape
    absDiffImage = abs(expectedImage - actualImage)

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

    _png.write_png(save_image_np.tostring(), width, height, output)
