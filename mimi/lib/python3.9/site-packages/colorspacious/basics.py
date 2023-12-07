# This file is part of colorspacious
# Copyright (C) 2014-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Basic colorspaces: conversions between sRGB, XYZ, xyY, CIELab

import numpy as np

from .util import stacklast, color_cart2polar, color_polar2cart
from .illuminants import as_XYZ100_w
from .testing import check_conversion

################################################################
# sRGB <-> sRGB-linear <-> XYZ100
################################################################

# https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation
def C_linear(C_srgb):
    out = np.empty(C_srgb.shape, dtype=float)
    linear_portion = (C_srgb < 0.04045)
    a = 0.055
    out[linear_portion] = C_srgb[linear_portion] / 12.92
    out[~linear_portion] = ((C_srgb[~linear_portion] + a) / (a + 1)) ** 2.4
    return out

def C_srgb(C_linear):
    out = np.empty(C_linear.shape, dtype=float)
    linear_portion = (C_linear <= 0.0031308)
    a = 0.055
    out[linear_portion] = C_linear[linear_portion] * 12.92
    out[~linear_portion] = (1+a) * C_linear[~linear_portion] ** (1/2.4) - a
    return out

XYZ100_to_sRGB1_matrix = np.array([
    # This is the exact matrix specified in IEC 61966-2-1:1999
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570],
    ])

# Condition number is 4.3, inversion is safe:
sRGB1_to_XYZ100_matrix = np.linalg.inv(XYZ100_to_sRGB1_matrix)

def XYZ100_to_sRGB1_linear(XYZ100):
    """Convert XYZ to linear sRGB, where XYZ is normalized so that reference
    white D65 is X=95.05, Y=100, Z=108.90 and sRGB is on the 0-1 scale. Linear
    sRGB has a linear relationship to actual light, so it is an appropriate
    space for simulating light (e.g. for alpha blending).

    """
    XYZ100 = np.asarray(XYZ100, dtype=float)
    # this is broadcasting matrix * array-of-vectors, where the vector is the
    # last dim
    RGB_linear = np.einsum("...ij,...j->...i", XYZ100_to_sRGB1_matrix, XYZ100 / 100)
    return RGB_linear

def sRGB1_linear_to_sRGB1(sRGB1_linear):
    return C_srgb(np.asarray(sRGB1_linear, dtype=float))

def sRGB1_to_sRGB1_linear(sRGB1):
    """Convert sRGB (as floats in the 0-to-1 range) to linear sRGB."""
    sRGB1 = np.asarray(sRGB1, dtype=float)
    sRGB1_linear = C_linear(sRGB1)
    return sRGB1_linear

def sRGB1_linear_to_XYZ100(sRGB1_linear):
    sRGB1_linear = np.asarray(sRGB1_linear, dtype=float)
    # this is broadcasting matrix * array-of-vectors, where the vector is the
    # last dim
    XYZ100 = np.einsum("...ij,...j->...i", sRGB1_to_XYZ100_matrix, sRGB1_linear)
    XYZ100 *= 100
    return XYZ100

def test_sRGB1_to_sRGB1_linear():
    from .gold_values import sRGB1_sRGB1_linear_gold
    check_conversion(sRGB1_to_sRGB1_linear, sRGB1_linear_to_sRGB1,
                     sRGB1_sRGB1_linear_gold,
                     a_max=1, b_max=1)

def test_sRGB1_linear_to_XYZ100():
    from .gold_values import sRGB1_linear_XYZ100_gold
    check_conversion(sRGB1_linear_to_XYZ100, XYZ100_to_sRGB1_linear,
                     sRGB1_linear_XYZ100_gold,
                     a_max=1, b_max=100)

################################################################
# XYZ <-> xyY
################################################################

# These functions work identically for both the 0-100 and 0-1 versions of
# XYZ/xyY.
def XYZ_to_xyY(XYZ):
    XYZ = np.asarray(XYZ, dtype=float)
    norm = np.sum(XYZ, axis=-1, keepdims=True)
    xy = XYZ[..., :2] / norm
    return np.concatenate((xy, XYZ[..., 1:2]), axis=-1)

def xyY_to_XYZ(xyY):
    xyY = np.asarray(xyY, dtype=float)
    x = xyY[..., 0]
    y = xyY[..., 1]
    Y = xyY[..., 2]
    X = Y / y * x
    Z = Y / y * (1 - x - y)
    return stacklast(X, Y, Z)

_XYZ100_to_xyY100_test_vectors = [
    ([10, 20, 30], [ 10. / 60,  20. / 60, 20]),
    ([99, 98,  3], [99. / 200, 98. / 200, 98]),
    ]

_XYZ1_to_xyY1_test_vectors = [
    ([0.10, 0.20, 0.30], [ 0.10 / 0.60,  0.20 / 0.60, 0.20]),
    ([0.99, 0.98, 0.03], [0.99 / 2.00, 0.98 / 2.00, 0.98]),
    ]

def test_XYZ_to_xyY():
    check_conversion(XYZ_to_xyY, xyY_to_XYZ,
                     _XYZ100_to_xyY100_test_vectors, b_max=[1, 1, 100])

    check_conversion(XYZ_to_xyY, xyY_to_XYZ,
                     _XYZ1_to_xyY1_test_vectors, b_max=[1, 1, 1])

################################################################
# XYZ100 <-> CIEL*a*b*
################################################################

# https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
def _f(t):
    out = np.empty(t.shape, dtype=float)
    linear_portion = (t < (6. / 29) ** 3)
    out[linear_portion] = ((1. / 3) * (29. / 6) ** 2 * t[linear_portion]
                           + 4. / 29)
    out[~linear_portion] = t[~linear_portion] ** (1. / 3)
    return out

def XYZ100_to_CIELab(XYZ100, XYZ100_w):
    XYZ100 = np.asarray(XYZ100, dtype=float)
    XYZ100_w = as_XYZ100_w(XYZ100_w)

    fXYZ100_norm = _f(XYZ100 / XYZ100_w)
    L = 116 * fXYZ100_norm[..., 1:2] - 16
    a = 500 * (fXYZ100_norm[..., 0:1] - fXYZ100_norm[..., 1:2])
    b = 200 * (fXYZ100_norm[..., 1:2] - fXYZ100_norm[..., 2:3])
    return np.concatenate((L, a, b), axis=-1)

def _finv(t):
    linear_portion = (t <= 6. / 29)
    out = np.select([linear_portion, ~linear_portion],
                    [3 * (6. / 29) ** 2 * (t - 4. / 29),
                     t ** 3])
    return out

def CIELab_to_XYZ100(CIELab, XYZ100_w):
    CIELab = np.asarray(CIELab, dtype=float)
    XYZ100_w = as_XYZ100_w(XYZ100_w)

    L = CIELab[..., 0]
    a = CIELab[..., 1]
    b = CIELab[..., 2]
    X_w = XYZ100_w[..., 0]
    Y_w = XYZ100_w[..., 1]
    Z_w = XYZ100_w[..., 2]

    l_piece = 1. / 116 * (L + 16)
    X = X_w * _finv(l_piece + 1. / 500 * a)
    Y = Y_w * _finv(l_piece)
    Z = Z_w * _finv(l_piece - 1. / 200 * b)

    return stacklast(X, Y, Z)

def test_XYZ100_to_CIELab():
    from .gold_values import XYZ100_CIELab_gold_D65, XYZ100_CIELab_gold_D50

    check_conversion(XYZ100_to_CIELab, CIELab_to_XYZ100,
                     XYZ100_CIELab_gold_D65,
                     # Stick to randomized values in the mid-range to avoid
                     # hitting negative luminances
                     b_min=[10, -30, -30], b_max=[90, 30, 30],
                     XYZ100_w="D65")

    check_conversion(XYZ100_to_CIELab, CIELab_to_XYZ100,
                     XYZ100_CIELab_gold_D50,
                     # Stick to randomized values in the mid-range to avoid
                     # hitting negative luminances
                     b_min=[10, -30, -30], b_max=[90, 30, 30],
                     XYZ100_w="D50")

    XYZ100_1 = np.asarray(XYZ100_CIELab_gold_D65[0][0])
    CIELab_1 = np.asarray(XYZ100_CIELab_gold_D65[0][1])

    XYZ100_2 = np.asarray(XYZ100_CIELab_gold_D50[1][0])
    CIELab_2 = np.asarray(XYZ100_CIELab_gold_D50[1][1])

    XYZ100_mixed = np.concatenate((XYZ100_1[np.newaxis, :],
                                   XYZ100_2[np.newaxis, :]))
    CIELab_mixed = np.concatenate((CIELab_1[np.newaxis, :],
                                   CIELab_2[np.newaxis, :]))

    XYZ100_w_mixed = np.row_stack((as_XYZ100_w("D65"), as_XYZ100_w("D50")))

    assert np.allclose(XYZ100_to_CIELab(XYZ100_mixed, XYZ100_w=XYZ100_w_mixed),
                       CIELab_mixed, rtol=0.001)
    assert np.allclose(CIELab_to_XYZ100(CIELab_mixed, XYZ100_w=XYZ100_w_mixed),
                       XYZ100_mixed, rtol=0.001)

################################################################
# CIELab <-> CIELCh
################################################################

def CIELab_to_CIELCh(CIELab):
    CIELab = np.asarray(CIELab)
    L = CIELab[..., 0]
    a = CIELab[..., 1]
    b = CIELab[..., 2]
    C, h = color_cart2polar(a, b)
    return stacklast(L, C, h)

def CIELCh_to_CIELab(CIELCh):
    CIELCh = np.asarray(CIELCh)
    L = CIELCh[..., 0]
    C = CIELCh[..., 1]
    h = CIELCh[..., 2]
    a, b = color_polar2cart(C, h)
    return stacklast(L, a, b)
