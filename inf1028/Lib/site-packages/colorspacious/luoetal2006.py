# This file is part of colorspacious
# Copyright (C) 2014 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

from .util import stacklast, color_cart2polar, color_polar2cart

class LuoEtAl2006UniformSpace(object):
    """A uniform space based on CIECAM02.

    See :cite:`CAM02-UCS` for details of the parametrization.

    For most purposes you should just use one of the predefined instances of
    this class that are exported as module-level constants:

    * :data:`colorspacious.CAM02UCS`
    * :data:`colorspacious.CAM02LCD`
    * :data:`colorspacious.CAM02SCD`
    """

    def __init__(self, KL, c1, c2):
        self.KL = KL
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        return "%s(%r, %r, %r)" % (
            self.__class__.__name__,
            self.KL, self.c1, self.c2)

    def JMh_to_Jpapbp(self, JMh):
        JMh = np.asarray(JMh, dtype=float)
        J = JMh[..., 0]
        M = JMh[..., 1]
        h = JMh[..., 2]
        Jp = (1 + 100 * self.c1) * J / (1 + self.c1 * J)
        Jp = Jp / self.KL
        Mp = (1. / self.c2) * np.log(1 + self.c2 * M)
        ap, bp = color_polar2cart(Mp, h)
        return stacklast(Jp, ap, bp)

    def Jpapbp_to_JMh(self, Jpapbp):
        Jpapbp = np.asarray(Jpapbp)
        Jp = Jpapbp[..., 0]
        ap = Jpapbp[..., 1]
        bp = Jpapbp[..., 2]
        Jp = Jp * self.KL
        J = - Jp / (self.c1 * Jp - 100 * self.c1 - 1)
        Mp, h = color_cart2polar(ap, bp)
        M = (np.exp(self.c2*Mp) - 1) / self.c2
        return stacklast(J, M, h)

CAM02UCS = LuoEtAl2006UniformSpace(1.00, 0.007, 0.0228)
CAM02LCD = LuoEtAl2006UniformSpace(0.77, 0.007, 0.0053)
CAM02SCD = LuoEtAl2006UniformSpace(1.24, 0.007, 0.0363)

def test_repr():
    # smoke test
    repr(CAM02UCS)
