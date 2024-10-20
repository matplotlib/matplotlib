# This file is part of colorspacious
# Copyright (C) 2014-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

from .conversion import cspace_convert

def deltaE(color1, color2,
           input_space="sRGB1", uniform_space="CAM02-UCS"):
    """Computes the :math:`\Delta E` distance between pairs of colors.

    :param input_space: The space the colors start out in. Can be anything
       recognized by :func:`cspace_convert`. Default: "sRGB1"
    :param uniform_space: Which space to perform the distance measurement
       in. This should be a uniform space like CAM02-UCS where
       Euclidean distance approximates similarity judgements, because
       otherwise the results of this function won't be very meaningful, but in
       fact any color space known to :func:`cspace_convert` will be accepted.

    By default, computes the euclidean distance in CAM02-UCS :math:`J'a'b'`
    space (thus giving :math:`\Delta E'`); for details, see
    :cite:`CAM02-UCS`. If you want the classic :math:`\Delta E^*_{ab}` defined
    by CIE 1976, use ``uniform_space="CIELab"``. Other good choices include
    ``"CAM02-LCD"`` and ``"CAM02-SCD"``.

    This function has no ability to perform :math:`\Delta E` calculations like
    CIEDE2000 that are not based on euclidean distances.

    This function is vectorized, i.e., color1, color2 may be arrays with shape
    (..., 3), in which case we compute the distance between corresponding
    pairs of colors.

    """

    uniform1 = cspace_convert(color1, input_space, uniform_space)
    uniform2 = cspace_convert(color2, input_space, uniform_space)
    return np.sqrt(np.sum((uniform1 - uniform2) ** 2, axis=-1))


def test_deltaE():
    from .gold_values import deltaE_1976_sRGB255_gold
    for a, b, gold in deltaE_1976_sRGB255_gold:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        got = deltaE(a, b, input_space="sRGB255", uniform_space="CIELab")
        np.testing.assert_allclose(got, gold, rtol=1e-4)

        got = deltaE(a / 255.0, b / 255.0,
                     input_space="sRGB1", uniform_space="CIELab")
        np.testing.assert_allclose(got, gold, rtol=1e-4)

        got = deltaE(a / 255.0, b / 255.0, uniform_space="CIELab")
        np.testing.assert_allclose(got, gold, rtol=1e-4)

    # Check vectorization
    all_a = [obj[0] for obj in deltaE_1976_sRGB255_gold]
    all_b = [obj[1] for obj in deltaE_1976_sRGB255_gold]
    all_gold = [obj[2] for obj in deltaE_1976_sRGB255_gold]
    got = deltaE(all_a, all_b, input_space="sRGB255", uniform_space="CIELab")
    np.testing.assert_allclose(got, all_gold, rtol=1e-4)

    from .gold_values import deltaE_sRGB255_CAM02UCS_silver
    for a, b, silver in deltaE_sRGB255_CAM02UCS_silver:
        got = deltaE(a, b, input_space="sRGB255")
        np.testing.assert_allclose(got, silver)
