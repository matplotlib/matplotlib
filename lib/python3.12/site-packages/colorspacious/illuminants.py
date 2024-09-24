# This file is part of colorspacious
# Copyright (C) 2014 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

__all__ = ["standard_illuminant_XYZ100", "as_XYZ100_w"]

# The "standard" 2 degree observer (CIE 1931)
# Sourced from http://www.easyrgb.com/index.php?X=MATH&H=15
cie1931 = {
    "A":   [109.850, 100,  35.585],
    "C":   [ 98.074, 100, 118.232],
    "D50": [ 96.422, 100,  82.521],
    "D55": [ 95.682, 100,  92.149],
    # This is sRGB's standard whitepoint:
    "D65": [ 95.047, 100, 108.883],
    "D75": [ 94.972, 100, 122.638],
}

# The "supplementary" 10 degree observer (CIE 1964)
# Sourced from http://www.easyrgb.com/index.php?X=MATH&H=15
cie1964 = {
    "A":   [111.144, 100,  35.200],
    "C":   [ 97.285, 100, 116.145],
    "D50": [ 96.720, 100,  81.427],
    "D55": [ 95.799, 100,  90.926],
    "D65": [ 94.811, 100, 107.304],
    "D75": [ 94.416, 100, 120.641],
}

# Fairchild (2005) says: "The difference between the two standard observers is
# significant, so care should be taken to report which observer is used with
# any colorimetric data. The differences are computationally significant, but
# certainly within the variability of color matching functions found for
# either 2 degree or 10 degree visual fields. Thus the two standard
# colorimetric observers can be thought of as representing the color matching
# functions of two individuals." (page 77)
#
# Though he also says: "It should also be noted that for sample sizes greater
# than 4 degrees, use of the CIE 1964 supplementary standard colorimetric
# observer is recommended." (page 82, when discussing CIE94 delta-E)

def standard_illuminant_XYZ100(name, observer="CIE 1931 2 deg"):
    """Takes a string naming a standard illuminant, and returns its XYZ
    coordinates (normalized to Y = 100).

    We currently have the following standard illuminants in our database:

    * ``"A"``
    * ``"C"``
    * ``"D50"``
    * ``"D55"``
    * ``"D65"``
    * ``"D75"``

    If you need another that isn't on this list, then feel free to send a pull
    request.

    When in doubt, use D65: it's the whitepoint used by the sRGB standard
    (61966-2-1:1999) and ISO 10526:1999 says "D65 should be used in all
    colorimetric calculations requiring representative daylight, unless there
    are specific reasons for using a different illuminant".

    By default, we return points in the XYZ space defined by the CIE 1931 2
    degree standard observer. By specifying ``observer="CIE 1964 10 deg"``,
    you can instead get the whitepoint coordinates in XYZ space defined by the
    CIE 1964 10 degree observer. This is probably only useful if you have XYZ
    points you want to do calculations on that were somehow measured using the
    CIE 1964 color matching functions, perhaps via a spectrophotometer;
    consumer equipment (monitors, cameras, etc.) assumes the use of the CIE
    1931 standard observer in all cases I know of.

    """
    if observer == "CIE 1931 2 deg":
        return np.asarray(cie1931[name], dtype=float)
    elif observer == "CIE 1964 10 deg":
        return np.asarray(cie1964[name], dtype=float)
    else:
        raise ValueError("observer must be 'CIE 1931 2 deg' or "
                         "'CIE 1964 10 deg', not %s" % (observer,))

def test_standard_illuminant_XYZ100():
    assert np.allclose(
        standard_illuminant_XYZ100("D65"),
        [ 95.047, 100, 108.883])

    assert np.allclose(
        standard_illuminant_XYZ100("D65", observer="CIE 1964 10 deg"),
        [ 94.811, 100, 107.304])

    from nose.tools import assert_raises
    assert_raises(ValueError, standard_illuminant_XYZ100, "D65",
                  observer="something else")

# Convenience function
def as_XYZ100_w(whitepoint):
    """A convenience function for getting whitepoints.

    ``whitepoint`` can be either a string naming a standard illuminant (see
    :func:`standard_illuminant_XYZ100`), or else a whitepoint given explicitly
    as an array-like of XYZ values.

    We internally call this function anywhere you have to specify a whitepoint
    (e.g. for CIECAM02 or CIELAB conversions).

    Always uses the "standard" 2 degree observer.

    """
    if isinstance(whitepoint, str):
        return standard_illuminant_XYZ100(whitepoint)
    else:
        whitepoint = np.asarray(whitepoint, dtype=float)
        if whitepoint.shape[-1] != 3:
            raise ValueError("Bad whitepoint shape")
        return whitepoint

def test_as_XYZ100_w():
    assert np.allclose(as_XYZ100_w("D65"), [ 95.047, 100, 108.883])
    assert np.allclose(as_XYZ100_w([1, 2, 3]), [1, 2, 3])
    assert as_XYZ100_w([1, 2, 3]).dtype == float

    from nose.tools import assert_raises
    assert_raises(KeyError, as_XYZ100_w, "D666")
    assert_raises(ValueError, as_XYZ100_w, [1, 2, 3, 4])
