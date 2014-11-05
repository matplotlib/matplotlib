#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for converting between color spaces.

Colorconv is copied from scikit-image to avoid an additional dependency on
scikit-image in the matplotlib documentation. You should almost sertanly use
the original module for any other use. This only contains the bare minumum
functions needed for rgb2lab Utility functions copied from dtype.py

The "central" color space in this module is RGB, more specifically the linear
sRGB color space using D65 as a white-point [1]_.  This represents a
standard monitor (w/o gamma correction). For a good FAQ on color spaces see
[2]_.

The API consists of functions to convert to and from RGB as defined above, as
well as a generic function to convert to and from any supported color space
(which is done through RGB in most cases).


Supported color spaces
----------------------
* RGB : Red Green Blue.
        Here the sRGB standard [1]_.
* HSV : Hue, Saturation, Value.
        Uniquely defined when related to sRGB [3]_.
* RGB CIE : Red Green Blue.
        The original RGB CIE standard from 1931 [4]_. Primary colors are 700 nm
        (red), 546.1 nm (blue) and 435.8 nm (green).
* XYZ CIE : XYZ
        Derived from the RGB CIE color space. Chosen such that
        ``x == y == z == 1/3`` at the whitepoint, and all color matching
        functions are greater than zero everywhere.
* LAB CIE : Lightness, a, b
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LUV CIE : Lightness, u, v
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LCH CIE : Lightness, Chroma, Hue
        Defined in terms of LAB CIE.  C and H are the polar representation of
        a and b.  The polar angle C is defined to be on ``(0, 2*pi)``

:author: Nicolas Pinto (rgb2hsv)
:author: Ralf Gommers (hsv2rgb)
:author: Travis Oliphant (XYZ and RGB CIE functions)
:author: Matt Terry (lab2lch)

:license: modified BSD

References
----------
.. [1] Official specification of sRGB, IEC 61966-2-1:1999.
.. [2] http://www.poynton.com/ColorFAQ.html
.. [3] http://en.wikipedia.org/wiki/HSL_and_HSV
.. [4] http://en.wikipedia.org/wiki/CIE_1931_color_space
"""

from __future__ import division

import numpy as np
from numpy import linalg

def _prepare_colorarray(arr):
    """Check the shape of the array and convert it to
    floating point representation.

    """
    arr = np.asanyarray(arr)

    if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
        msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
               "got (" + (", ".join(map(str, arr.shape))) + ")")
        raise ValueError(msg)

    return img_as_float(arr)


# ---------------------------------------------------------------
# Primaries for the coordinate systems
# ---------------------------------------------------------------
cie_primaries = np.array([700, 546.1, 435.8])
sb_primaries = np.array([1. / 155, 1. / 190, 1. / 225]) * 1e5

# ---------------------------------------------------------------
# Matrices that define conversion between different color spaces
# ---------------------------------------------------------------

# From sRGB specification
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = linalg.inv(xyz_from_rgb)
# XYZ coordinates of the illuminants, scaled to [0, 1]. For each illuminant I
# we have:
#
#   illuminant[I][0] corresponds to the XYZ coordinates for the 2 degree
#   field of view.
#
#   illuminant[I][1] corresponds to the XYZ coordinates for the 10 degree
#   field of view.
#
# The XYZ coordinates are calculated from [1], using the formula:
#
#   X = x * ( Y / y )
#   Y = Y
#   Z = ( 1 - x - y ) * ( Y / y )
#
# where Y = 1. The only exception is the illuminant "D65" with aperture angle
# 2, whose coordinates are copied from 'lab_ref_white' for
# backward-compatibility reasons.
#
#     References
#    ----------
#    .. [1] http://en.wikipedia.org/wiki/Standard_illuminant

illuminants = \
    {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
           '10': (1.111420406956693, 1, 0.3519978321919493)},
     "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
             '10': (0.9672062750333777, 1, 0.8142801513128616)},
     "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
             '10': (0.9579665682254781, 1, 0.9092525159847462)},
     "D65": {'2': (0.95047, 1., 1.08883),   # This was: `lab_ref_white`
             '10': (0.94809667673716, 1, 1.0730513595166162)},
     "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
             '10': (0.9441713925645873, 1, 1.2064272211720228)},
     "E": {'2': (1.0, 1.0, 1.0),
           '10': (1.0, 1.0, 1.0)}}


def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.

    Parameters
    ----------
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.

    Returns
    -------
    (x, y, z) : tuple
        A tuple with 3 elements containing the XYZ coordinates of the given
        illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Standard_illuminant

    """
    illuminant = illuminant.upper()
    try:
        return illuminants[illuminant][observer]
    except KeyError:
        raise ValueError("Unknown illuminant/observer combination\
        (\'{0}\', \'{1}\')".format(illuminant, observer))


def _convert(matrix, arr):
    """Do the color space conversion.

    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.

    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """
    arr = _prepare_colorarray(arr)
    arr = np.swapaxes(arr, 0, -1)
    oldshape = arr.shape
    arr = np.reshape(arr, (3, -1))
    out = np.dot(matrix, arr)
    out.shape = oldshape
    out = np.swapaxes(out, -1, 0)

    return np.ascontiguousarray(out)


def rgb2xyz(rgb):
    """RGB to XYZ color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.

    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.

    Raises
    ------
    ValueError
        If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space


    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _prepare_colorarray(rgb).copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return _convert(xyz_from_rgb, arr)


def xyz2lab(xyz, illuminant="D65", observer="2"):
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : ndarray
        The image in CIE-LAB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.

    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape ``(.., ..,[ ..,] 3)``.
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.

    Notes
    -----
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] http://en.wikipedia.org/wiki/Lab_color_space

    """
    arr = _prepare_colorarray(xyz)

    xyz_ref_white = get_xyz_coords(illuminant, observer)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.power(arr[mask], 1. / 3.)
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape ``(.., .., 3)``.

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/CIE_1931_color_space


    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return arr


def rgb2lab(rgb):
    """RGB to lab color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.

    Returns
    -------
    out : ndarray
        The image in Lab format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.

    Raises
    ------
    ValueError
        If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.

    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    """
    return xyz2lab(rgb2xyz(rgb))


def lab2rgb(lab):
    """Lab to RGB color space conversion.

    Parameters
    ----------
    lab : array_like
        The image in Lab format, in a 3-D array of shape ``(.., .., 3)``.

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.

    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    """
    return xyz2rgb(lab2xyz(lab))


def lab2xyz(lab, illuminant="D65", observer="2"):
    """CIE-LAB to XYZcolor space conversion.

    Parameters
    ----------
    lab : array_like
        The image in lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.


    Notes
    -----
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values x_ref
    = 95.047, y_ref = 100., z_ref = 108.883. See function 'get_xyz_coords' for
    a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] http://en.wikipedia.org/wiki/Lab_color_space

    """

    arr = _prepare_colorarray(lab).copy()

    L, a, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    out = np.dstack([x, y, z])

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    out *= xyz_ref_white
    return out


def convert(image, dtype, force_copy=False, uniform=False):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.

    References
    ----------
    (1) DirectX data conversion rules.
        http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    (2) Data Conversions.
        In "OpenGL ES 2.0 Specification v2.0.25", pp 7-8. Khronos Group, 2010.
    (3) Proper treatment of pixels as integers. A.W. Paeth.
        In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    (4) Dirty Pixels. J. Blinn.
        In "Jim Blinn's corner: Dirty Pixels", pp 47-57. Morgan Kaufmann, 1998.

    """
    image = np.asarray(image)
    dtypeobj = np.dtype(dtype)
    dtypeobj_in = image.dtype
    dtype = dtypeobj.type
    dtype_in = dtypeobj_in.type

    if dtype_in == dtype:
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype in _supported_types):
        raise ValueError("can not convert %s to %s." % (dtypeobj_in, dtypeobj))

    def sign_loss():
        warn("Possible sign loss when converting negative image of type "
             "%s to positive image of type %s." % (dtypeobj_in, dtypeobj))

    def prec_loss():
        warn("Possible precision loss when converting from "
             "%s to %s" % (dtypeobj_in, dtypeobj))

    def _dtype(itemsize, *dtypes):
        # Return first of `dtypes` with itemsize greater than `itemsize`
        return next(dt for dt in dtypes if itemsize < np.dtype(dt).itemsize)

    def _dtype2(kind, bits, itemsize=1):
        # Return dtype of `kind` that can store a `bits` wide unsigned int
        c = lambda x, y: x <= y if kind == 'u' else x < y
        s = next(i for i in (itemsize, ) + (2, 4, 8) if c(bits, i * 8))
        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        # Scale unsigned/positive integers from n to m bits
        # Numbers can be represented exactly only if m is a multiple of n
        # Output array is of same kind as input.
        kind = a.dtype.kind
        if n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            prec_loss()
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                                casting='unsafe')
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of n bits
            if copy:
                b = np.empty(a.shape, _dtype2(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = np.array(a, _dtype2(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of n bits,
            # then downscale with precision loss
            prec_loss()
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype2(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2**(o - m)
                return b
            else:
                a = np.array(a, _dtype2(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2**(o - m)
                return a

    kind = dtypeobj.kind
    kind_in = dtypeobj_in.kind
    itemsize = dtypeobj.itemsize
    itemsize_in = dtypeobj_in.itemsize

    if kind == 'b':
        # to binary image
        if kind_in in "fi":
            sign_loss()
        prec_loss()
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    if kind_in == 'b':
        # from binary image, to float and to integer
        result = image.astype(dtype)
        if kind != 'f':
            result *= dtype(dtype_range[dtype][1])
        return result

    if kind in 'ui':
        imin = np.iinfo(dtype).min
        imax = np.iinfo(dtype).max
    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max

    if kind_in == 'f':
        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        if kind == 'f':
            # floating point -> floating point
            if itemsize_in > itemsize:
                prec_loss()
            return image.astype(dtype)

        # floating point -> integer
        prec_loss()
        # use float type that can represent output integer type
        image = np.array(image, _dtype(itemsize, dtype_in,
                                       np.float32, np.float64))
        if not uniform:
            if kind == 'u':
                image *= imax
            else:
                image *= imax - imin
                image -= 1.0
                image /= 2.0
            np.rint(image, out=image)
            np.clip(image, imin, imax, out=image)
        elif kind == 'u':
            image *= imax + 1
            np.clip(image, 0, imax, out=image)
        else:
            image *= (imax - imin + 1.0) / 2.0
            np.floor(image, out=image)
            np.clip(image, imin, imax, out=image)
        return image.astype(dtype)

    if kind == 'f':
        # integer -> floating point
        if itemsize_in >= itemsize:
            prec_loss()
        # use float type that can exactly represent input integers
        image = np.array(image, _dtype(itemsize_in, dtype,
                                       np.float32, np.float64))
        if kind_in == 'u':
            image /= imax_in
            # DirectX uses this conversion also for signed ints
            #if imin_in:
            #    np.maximum(image, -1.0, out=image)
        else:
            image *= 2.0
            image += 1.0
            image /= imax_in - imin_in
        return image.astype(dtype)

    if kind_in == 'u':
        if kind == 'i':
            # unsigned integer -> signed integer
            image = _scale(image, 8 * itemsize_in, 8 * itemsize - 1)
            return image.view(dtype)
        else:
            # unsigned integer -> unsigned integer
            return _scale(image, 8 * itemsize_in, 8 * itemsize)

    if kind == 'u':
        # signed integer -> unsigned integer
        sign_loss()
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize)
        result = np.empty(image.shape, dtype)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed integer -> signed integer
    if itemsize_in > itemsize:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize - 1)
    image = image.astype(_dtype2('i', itemsize * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize, copy=False)
    image += imin
    return image.astype(dtype)


def img_as_float(image, force_copy=False):
    """Convert an image to double-precision floating point format.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float64
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.

    """
    return convert(image, np.float64, force_copy)
