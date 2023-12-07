# This file is part of colorspacious
# Copyright (C) 2014 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

from __future__ import division

from collections import namedtuple

import numpy as np

from .illuminants import as_XYZ100_w

__all__ = [
    "CIECAM02Surround", "CIECAM02Space", "NegativeAError",
    "JChQMsH",
]


# F, c, Nc: surround parameters
#            F     c      Nc
# Average   1.0   0.69   1.0
# Dim       0.9   0.59   0.9
# Dark      0.8   0.525  0.8
# See: https://github.com/njsmith/colorspacious/issues/14
CIECAM02Surround = namedtuple("CIECAM02Surround", ["F", "c", "N_c"])
CIECAM02Surround.AVERAGE = CIECAM02Surround(1.0, 0.69,  1.0)
CIECAM02Surround.DIM     = CIECAM02Surround(0.9, 0.59,  0.9)
CIECAM02Surround.DARK    = CIECAM02Surround(0.8, 0.525, 0.8)

JChQMsH = namedtuple("JChQMsH", ["J", "C", "h", "Q", "M", "s", "H"])

M_CAT02 = np.asarray([[ 0.7328,  0.4296, -0.1624],
                      [-0.7036,  1.6975,  0.0061],
                      [ 0.0030,  0.0136,  0.9834]])

M_HPE = np.asarray([[ 0.38971,  0.68898, -0.07868],
                    [-0.22981,  1.18340,  0.04641],
                    [ 0.00000,  0.00000,  1.00000]])

# These are very well-conditioned matrices (condition numbers <4), so just
# taking the inverse is fine, and it simplifies things below.
M_CAT02_inv = np.linalg.inv(M_CAT02)
M_HPE_M_CAT02_inv = np.dot(M_HPE, M_CAT02_inv)
M_CAT02_M_HPE_inv = np.dot(M_CAT02, np.linalg.inv(M_HPE))

h_i = np.asarray([20.14,  90.00, 164.25, 237.53, 380.14])
e_i = np.asarray([ 0.8,    0.7,    1.0,    1.2,    0.8])
H_i = np.asarray([ 0.0,  100.0,  200.0,  300.0,  400.0])

def broadcasting_matvec(A, B):
    # We want to handle two cases that come up a bunch below.
    # B is always a vector, or collection of vectors. So it has shape
    #    (..., j)
    # where j is the number of entries in each vector.
    # A is either a matrix or a vector, and we want to broadcast np.dot(A,
    # B_vec) over all possible B_vecs.
    # When A has shape (i, j), this means our result should have shape
    #    (..., i)
    # and when A has shape (j,), this means our result should have shape
    #    (...)
    # So a generalization is that given
    #   A.shape == (...1, j)
    #   B.shape == (...2, j)
    # we want a result with shape
    #   (...2, ...1)
    # it turns out that this is the magic incantation for doing that:
    return np.inner(B, A)

def require_exactly_one(**kwargs):
    non_None = sum(v is not None for v in kwargs.values())
    if non_None != 1:
        raise ValueError("Exactly one of %s must be specified"
                         % (", ".join(kwargs)))

class NegativeAError(ValueError):
    """A :class:`ValueError` that can be raised when converting to CIECAM02.

    See :meth:`CIECAM02Space.XYZ100_to_CIECAM02` for details.
    """
    pass

class CIECAM02Space(object):
    """An object representing a particular set of CIECAM02 viewing conditions.

    :param XYZ100_w: The whitepoint. Either a string naming one of the known
         standard whitepoints like ``"D65"``, or else a point in XYZ100 space.
    :param Y_b: Background luminance.
    :param L_A: Luminance of the adapting field (in cd/m^2).
    :param surround: A :class:`CIECAM02Surround` object.

    """
    def __init__(self, XYZ100_w, Y_b, L_A,
                 surround=CIECAM02Surround.AVERAGE):
        self.XYZ100_w = as_XYZ100_w(XYZ100_w)
        # as_XYZ100_w allows for multiple whitepoints to be returned, but we
        # aren't vectorized WRT whitepoint
        if self.XYZ100_w.shape != (3,):
            raise ValueError("Hey! XYZ100_w should have shape (3,)!")
        self.Y_b = float(Y_b)
        self.L_A = float(L_A)
        self.surround = surround
        self.F = float(surround.F)
        self.c = float(surround.c)
        self.N_c = float(surround.N_c)

        self.RGB_w = np.dot(M_CAT02, self.XYZ100_w)
        self.D = self.F * (1 - (1/3.6) * np.exp((-self.L_A - 42) / 92))
        self.D = np.clip(self.D, 0, 1)

        self.D_RGB = self.D * self.XYZ100_w[1] / self.RGB_w + 1 - self.D
        # Fairchild (2013), pages 290-292, recommends using this equation
        # instead, though notes that it doesn't make much difference as part
        # of a full CIECAM02 system. (It matters more if you're only using
        # pieces.)
        #self.D_RGB = self.D * 100 / self.RGB_w + 1 - self.D
        self.k = 1 / (5 * self.L_A + 1)
        self.F_L = (0.2 * self.k ** 4 * (5 * self.L_A)
                    + 0.1 * (1 - self.k**4)**2 * (5 * self.L_A) ** (1./3))
        self.n = self.Y_b / self.XYZ100_w[1]
        self.z = 1.48 + np.sqrt(self.n)
        self.N_bb = 0.725 * (1 / self.n)**0.2
        self.N_cb = self.N_bb  #??

        self.RGB_wc = self.D_RGB * self.RGB_w
        self.RGBprime_w = np.dot(M_HPE_M_CAT02_inv, self.RGB_wc)
        tmp = ((self.F_L * self.RGBprime_w) / 100) ** 0.42
        self.RGBprime_aw = 400 * (tmp / (tmp + 27.13)) + 0.1
        self.A_w = ((np.dot([2, 1, 1. / 20], self.RGBprime_aw) - 0.305)
                    * self.N_bb)

    def __repr__(self):
        surround_string = ", surround=%r" % (self.surround,)
        if self.surround == CIECAM02Surround.AVERAGE:
            surround_string = ""
        return "%s(%r, %r, %r%s) " % (
            self.__class__.__name__,
            list(self.XYZ100_w),
            self.Y_b,
            self.L_A,
            surround_string)

    # XYZ100 must have shape (3,) or (3, n)
    def XYZ100_to_CIECAM02(self, XYZ100, on_negative_A="raise"):
        """Computes CIECAM02 appearance correlates for the given tristimulus
        value(s) XYZ (normalized to be on the 0-100 scale).

        Example: ``vc.XYZ100_to_CIECAM02([30.0, 45.5, 21.0])``

        :param XYZ100: An array-like of tristimulus values. These should be
          given on the 0-100 scale, not the 0-1 scale. The array-like should
          have shape ``(..., 3)``; e.g., you can use a simple 3-item list
          (shape = ``(3,)``), or to efficiently perform multiple computations
          at once, you could pass a higher-dimensional array, e.g. an image.
        :arg on_negative_A: A known infelicity of the CIECAM02 model is that
          for some inputs, the achromatic signal :math:`A` can be negative,
          which makes it impossible to compute :math:`J`, :math:`C`,
          :math:`Q`, :math:`M`, or :math:`s` -- only :math:`h`: and :math:`H`
          are spared. (See, e.g., section 2.6.4.1 of :cite:`Luo-CIECAM02` for
          discussion.) This argument allows you to specify a strategy for
          handling such points. Options are:

          * ``"raise"``: throws a :class:`NegativeAError` (a subclass of
            :class:`ValueError`)
          * ``"nan"``: return not-a-number values for the affected
            elements. (This may be particularly useful if converting a large
            number of points at once.)

        :returns: A named tuple of type :class:`JChQMsH`, with attributes
          ``J``, ``C``, ``h``, ``Q``, ``M``, ``s``, and ``H`` containing the
          CIECAM02 appearance correlates.

        """

        #### Argument checking

        XYZ100 = np.asarray(XYZ100, dtype=float)
        if XYZ100.shape[-1] != 3:
            raise ValueError("XYZ100 shape must be (..., 3)")

        #### Step 1

        RGB = broadcasting_matvec(M_CAT02, XYZ100)

        #### Step 2

        RGB_C = self.D_RGB * RGB

        #### Step 3

        RGBprime = broadcasting_matvec(M_HPE_M_CAT02_inv, RGB_C)

        #### Step 4

        RGBprime_signs = np.sign(RGBprime)

        tmp = (self.F_L * RGBprime_signs * RGBprime / 100) ** 0.42
        RGBprime_a = RGBprime_signs * 400 * (tmp / (tmp + 27.13)) + 0.1

        #### Step 5

        a = broadcasting_matvec([1, -12. / 11, 1. / 11], RGBprime_a)
        b = broadcasting_matvec([1. / 9, 1. / 9, -2. / 9], RGBprime_a)
        h_rad = np.arctan2(b, a)
        h = np.rad2deg(h_rad) % 360

        # #### Step 6

        # hprime = h, unless h < 20.14, in which case hprime = h + 360.
        hprime = np.select([h < h_i[0], True], [h + 360, h])
        # we use 0-based indexing, so our i is one less than the reference
        # formulas' i.
        i = np.searchsorted(h_i, hprime, side="right") - 1
        tmp = (hprime - h_i[i]) / e_i[i]
        H = H_i[i] + ((100 * tmp)
                      / (tmp + (h_i[i + 1] - hprime) / e_i[i + 1]))

        #### Step 7

        A = ((broadcasting_matvec([2, 1, 1. / 20], RGBprime_a) - 0.305)
             * self.N_bb)

        if on_negative_A == "raise":
            if np.any(A < 0):
                raise NegativeAError("attempted to convert a tristimulus "
                                     "value whose achromatic signal was "
                                     "negative, and on_negative_A=\"raise\"")
        elif on_negative_A == "nan":
            A = np.select([A < 0, True], [np.nan, A])
        else:
            raise ValueError("Invalid on_negative_A argument: got %r, "
                             "expected \"raise\" or \"nan\""
                             % (on_negative_A,))

        #### Step 8

        J = 100 * (A / self.A_w) ** (self.c * self.z)

        #### Step 9

        Q = self._J_to_Q(J)

        #### Step 10

        e = (12500. / 13) * self.N_c * self.N_cb * (np.cos(h_rad + 2) + 3.8)
        t = (e * np.sqrt(a ** 2 + b ** 2)
             / broadcasting_matvec([1, 1, 21. / 20], RGBprime_a))

        C = t**0.9 * (J / 100)**0.5 * (1.64 - 0.29**self.n)**0.73
        M = C * self.F_L**0.25
        s = 100 * (M / Q)**0.5

        return JChQMsH(J, C, h, Q, M, s, H)

    def _J_to_Q(self, J):
        return ((4 / self.c) * (J / 100) ** 0.5
                * (self.A_w + 4) * self.F_L**0.25)

    def CIECAM02_to_XYZ100(self, J=None, C=None, h=None,
                           Q=None, M=None, s=None, H=None):
        """Return the unique tristimulus values that have the given CIECAM02
        appearance correlates under these viewing conditions.

        You must specify 3 arguments:

        * Exactly one of ``J`` and ``Q``
        * Exactly one of ``C``, ``M``, and ``s``
        * Exactly one of ``h`` and ``H``.

        Arguments can be vectors, in which case they will be broadcast against
        each other.

        Returned tristimulus values will be on the 0-100 scale, not the 0-1
        scale.
        """

        #### Argument checking

        require_exactly_one(J=J, Q=Q)
        require_exactly_one(C=C, M=M, s=s)
        require_exactly_one(h=h, H=H)

        if J is not None:
            J = np.asarray(J, dtype=float)
        if C is not None:
            C = np.asarray(C, dtype=float)
        if h is not None:
            h = np.asarray(h, dtype=float)
        if Q is not None:
            Q = np.asarray(Q, dtype=float)
        if M is not None:
            M = np.asarray(M, dtype=float)
        if s is not None:
            s = np.asarray(s, dtype=float)
        if H is not None:
            H = np.asarray(H, dtype=float)

        #### Step 1: conversions to get JCh

        if J is None:
            J = 6.25 * ((self.c * Q) / ((self.A_w + 4) * self.F_L**0.25)) ** 2

        if C is None:
            if M is not None:
                C = M / self.F_L**0.25
            else:
                assert s is not None
                # when starting from s, we need Q
                if Q is None:
                    Q = self._J_to_Q(J)
                C = (s / 100) ** 2 * (Q / self.F_L**0.25)

        if h is None:
            i = np.searchsorted(H_i, H, side="right") - 1
            # BROKEN:
            num1 = (H - H_i[i]) * (e_i[i + 1] * h_i[i] - e_i[i] * h_i[i + 1])
            num2 = -100 * h_i[i] * e_i[i + 1]
            denom1 = (H - H_i[i]) * (e_i[i + 1] - e_i[i])
            denom2 = -100 * e_i[i + 1]
            hprime = (num1 + num2) / (denom1 + denom2)
            h = np.select([hprime > 360, True], [hprime - 360, hprime])

        J, C, h = np.broadcast_arrays(J, C, h)
        target_shape = J.shape

        # 0d arrays break indexing stuff
        if J.ndim == 0:
            J = np.atleast_1d(J)
            C = np.atleast_1d(C)
            h = np.atleast_1d(h)

        #### Step 2

        t = (C
             / (np.sqrt(J / 100) * (1.64 - 0.29**self.n) ** 0.73)
            ) ** (1 / 0.9)
        e_t = 0.25 * (np.cos(np.deg2rad(h) + 2) + 3.8)
        A = self.A_w * (J / 100) ** (1 / (self.c * self.z))

        # an awkward way of calculating 1/t such that 1/0 -> inf
        with np.errstate(divide="ignore"):
            one_over_t = 1 / t
        one_over_t = np.select([np.isnan(one_over_t), True],
                               [np.inf, one_over_t])

        p_1 = (50000. / 13) * self.N_c * self.N_cb * e_t * one_over_t
        p_2 = A / self.N_bb + 0.305
        p_3 = 21. / 20

        #### Step 3

        sin_h = np.sin(np.deg2rad(h))
        cos_h = np.cos(np.deg2rad(h))

        # to avoid divide-by-zero (or divide-by-eps) issues, we use different
        # computations when |sin_h| > |cos_h| and vice-versa
        num = p_2 * (2 + p_3) * (460. / 1403)
        denom_part2 = (2 + p_3) * (220. / 1403)
        denom_part3 = (-27. / 1403) + p_3 * (6300. / 1403)

        a = np.empty_like(h)
        b = np.empty_like(h)

        small_cos = (np.abs(sin_h) >= np.abs(cos_h))

        # NB denom_part2 and denom_part3 are scalars
        b[small_cos] = (num[small_cos]
                        / (p_1[small_cos] / sin_h[small_cos]
                           + (denom_part2
                              * cos_h[small_cos] / sin_h[small_cos])
                           + denom_part3))
        a[small_cos] = b[small_cos] * cos_h[small_cos] / sin_h[small_cos]

        a[~small_cos] = (num[~small_cos]
                         / (p_1[~small_cos] / cos_h[~small_cos]
                            + denom_part2
                            + (denom_part3
                               * sin_h[~small_cos] / cos_h[~small_cos])))
        b[~small_cos] = a[~small_cos] * sin_h[~small_cos] / cos_h[~small_cos]

        #### Step 4

        p2ab = np.concatenate((p_2[..., np.newaxis],
                               a[..., np.newaxis],
                               b[..., np.newaxis]),
                              axis=-1)
        RGBprime_a_matrix = (1. / 1403
                             * np.asarray([[ 460,  451,   288],
                                           [ 460, -891,  -261],
                                           [ 460, -220, -6300]], dtype=float))

        RGBprime_a = broadcasting_matvec(RGBprime_a_matrix, p2ab)

        #### Step 5

        RGBprime = (np.sign(RGBprime_a - 0.1)
                    * (100 / self.F_L)
                    * ((27.13 * np.abs(RGBprime_a - 0.1))
                       / (400 - np.abs(RGBprime_a - 0.1))) ** (1 / 0.42))

        #### Step 6

        RGB_C = broadcasting_matvec(M_CAT02_M_HPE_inv, RGBprime)

        #### Step 7

        RGB = RGB_C / self.D_RGB

        #### Step 8

        XYZ100 = broadcasting_matvec(M_CAT02_inv, RGB)

        XYZ100 = XYZ100.reshape(target_shape + (3,))

        return XYZ100

CIECAM02Space.sRGB = CIECAM02Space(
    # sRGB specifies a D65 monitor and a D50 ambient. CIECAM02 doesn't really
    # know how to deal with this discrepancy; it appears that the usual thing
    # to do is just to use D65 for the whitepoint.
    XYZ100_w="D65",
    Y_b=20,
    # To compute L_A:
    #   illuminance in lux / pi = luminance in cd/m^2
    #   luminance in cd/m^2 / 5 = L_A (the "grey world assumption")
    # See Moroney (2000), "Usage guidelines for CIECAM97s".
    # sRGB illuminance is 64 lux.
    L_A=(64 / np.pi) / 5,
    surround=CIECAM02Surround.AVERAGE)


################################################################
# Tests
################################################################

def check_roundtrip(vc, XYZ100):
    try:
        values = vc.XYZ100_to_CIECAM02(XYZ100, on_negative_A="raise")
    except NegativeAError:
        # don't expect to be able to round-trip these values
        return
    for kwarg1 in ["J", "Q"]:
        for kwarg2 in ["C", "M", "s"]:
            for kwarg3 in ["h", "H"]:
                got = vc.CIECAM02_to_XYZ100(**{kwarg1: getattr(values, kwarg1),
                                            kwarg2: getattr(values, kwarg2),
                                            kwarg3: getattr(values, kwarg3)})
                assert np.allclose(got, XYZ100)

def test_gold():
    from .gold_values import XYZ100_CIECAM02_gold

    for t in XYZ100_CIECAM02_gold:
        got = t.vc.XYZ100_to_CIECAM02(t.XYZ100)
        for i in range(len(got)):
            assert np.allclose(got[i], t.expected[i], atol=1e-05)
        check_roundtrip(t.vc, t.XYZ100)

def test_inverse():
    r = np.random.RandomState(0)
    XYZ100_values = [[0, 0, 0], [50, 50, 50]]
    for i in range(10):
        XYZ100_values.append(r.uniform(high=100, size=3))

    for illuminant in ["C", "D50", "D65"]:
        for Y_b in [20, 18]:
            for L_A in [30, 300]:
                for surround in [CIECAM02Surround.AVERAGE,
                                 CIECAM02Surround.DIM,
                                 CIECAM02Surround.DARK]:
                    vc = CIECAM02Space(illuminant, Y_b, L_A, surround)
                    for XYZ100 in XYZ100_values:
                        check_roundtrip(vc, XYZ100)

def test_misc():
    from nose.tools import assert_raises
    # Only one whitepoint can be specified
    assert_raises(ValueError, CIECAM02Space,
                  [[20, 100, 80], [80, 100, 20]], 20, 30)

    # smoke test
    repr(CIECAM02Space.sRGB)
    repr(CIECAM02Space("D65", 20, 4, surround=CIECAM02Surround.DIM))

    # input shape check
    assert_raises(ValueError,
                  CIECAM02Space.sRGB.XYZ100_to_CIECAM02,
                  np.ones((10, 4)))

    # on_negative_A validity check
    assert_raises(ValueError,
                  CIECAM02Space.sRGB.XYZ100_to_CIECAM02,
                  np.ones((10, 3)),
                  on_negative_A="asdfasdf")

def test_exactly_one():
    from nose.tools import assert_raises
    vc = CIECAM02Space.sRGB

    # Redundant specifications not allowed
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, C=1, h=1, Q=1)
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, C=1, h=1, M=1)
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, C=1, h=1, s=1)
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, C=1, h=1, H=1)

    # Underspecified colors not allowed either
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, C=1)
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, J=1, h=1)
    assert_raises(ValueError, vc.CIECAM02_to_XYZ100, C=1, h=1)

def test_vectorized():
    vc = CIECAM02Space.sRGB

    XYZ100s = [[20, 30, 40], [40, 30, 20]]
    CIECAM02s = vc.XYZ100_to_CIECAM02(XYZ100s)

    for i, XYZ100 in enumerate(XYZ100s):
        CIECAM02 = vc.XYZ100_to_CIECAM02(XYZ100)
        for j in range(len(CIECAM02)):
            assert np.allclose(CIECAM02[j], CIECAM02s[j][i])

    check_roundtrip(vc, XYZ100s)

def test_on_negative_A():
    from nose.tools import assert_raises

    vc = CIECAM02Space("D65", 20, 30)
    bad_XYZ100 = [8.71292997, 2.02183974, 83.26198455]
    good_XYZ100 = [20, 30, 40]

    assert_raises(NegativeAError, vc.XYZ100_to_CIECAM02, bad_XYZ100)
    assert_raises(NegativeAError, vc.XYZ100_to_CIECAM02,
                  [bad_XYZ100, good_XYZ100])
    assert_raises(NegativeAError, vc.XYZ100_to_CIECAM02, bad_XYZ100,
                  on_negative_A="raise")
    assert_raises(NegativeAError, vc.XYZ100_to_CIECAM02,
                  [bad_XYZ100, good_XYZ100],
                  on_negative_A="raise")

    bad_CIECAM02 = vc.XYZ100_to_CIECAM02(bad_XYZ100, on_negative_A="nan")
    for bad_attr in "JCQMs":
        assert np.isnan(getattr(bad_CIECAM02, bad_attr))
    assert np.allclose(bad_CIECAM02.h, 205.80008)
    assert np.allclose(bad_CIECAM02.H, 261.11054)

    mixed_CIECAM02 = vc.XYZ100_to_CIECAM02([bad_XYZ100, good_XYZ100],
                                           on_negative_A="nan")
    for bad_attr in "JCQMs":
        assert np.all(np.isnan(getattr(mixed_CIECAM02, bad_attr))
                      == [True, False])
    assert np.all(np.isnan(mixed_CIECAM02.h) == [False, False])
    assert np.all(np.isnan(mixed_CIECAM02.H) == [False, False])
