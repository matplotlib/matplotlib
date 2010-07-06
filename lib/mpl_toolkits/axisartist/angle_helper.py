from math import floor

import numpy as np
import math

A = np.array

from grid_finder import ExtremeFinderSimple

def select_step_degree(dv):

    degree_limits_ = [1.5, 3, 7, 13, 20, 40, 70, 120, 270, 520]
    degree_steps_  = [  1, 2, 5, 10, 15, 30, 45,  90, 180, 360]
    degree_factors = [1.] * len(degree_steps_)

    minsec_limits_ = [1.5, 2.5, 3.5, 8, 11, 18, 25, 45]
    minsec_steps_  = [1,   2,   3,   5, 10, 15, 20, 30]

    minute_limits_ = A(minsec_limits_)*(1./60.)
    minute_factors = [60.] * len(minute_limits_)

    second_limits_ = A(minsec_limits_)*(1./3600.)
    second_factors = [3600.] * len(second_limits_)

    degree_limits = np.concatenate([second_limits_,
                                    minute_limits_,
                                    degree_limits_])

    degree_steps = np.concatenate([minsec_steps_,
                                   minsec_steps_,
                                   degree_steps_])

    degree_factors = np.concatenate([second_factors,
                                     minute_factors,
                                     degree_factors])

    n = degree_limits.searchsorted(dv)
    step = degree_steps[n]
    factor = degree_factors[n]

    return step, factor



def select_step_hour(dv):

    hour_limits_ = [1.5, 2.5, 3.5, 5, 7, 10, 15, 21, 36]
    hour_steps_  = [1,   2  , 3,   4, 6,  8, 12, 18, 24]
    hour_factors = [1.] * len(hour_steps_)

    minsec_limits_ = [1.5, 2.5, 3.5, 4.5, 5.5, 8, 11, 14, 18, 25, 45]
    minsec_steps_  = [1,   2,   3,   4,   5,   6, 10, 12, 15, 20, 30]

    minute_limits_ = A(minsec_limits_)*(1./60.)
    minute_factors = [60.] * len(minute_limits_)

    second_limits_ = A(minsec_limits_)*(1./3600.)
    second_factors = [3600.] * len(second_limits_)

    hour_limits = np.concatenate([second_limits_,
                                  minute_limits_,
                                  hour_limits_])

    hour_steps = np.concatenate([minsec_steps_,
                                 minsec_steps_,
                                 hour_steps_])

    hour_factors = np.concatenate([second_factors,
                                   minute_factors,
                                   hour_factors])

    n = hour_limits.searchsorted(dv)
    step = hour_steps[n]
    factor = hour_factors[n]

    return step, factor


def select_step_sub(dv):

    # subarcsec or degree
    tmp = 10.**(int(math.log10(dv))-1.)
    dv2 = dv/tmp
    substep_limits_ = [1.5, 3., 7.]
    substep_steps_  = [1. , 2., 5.]

    factor = 1./tmp

    if 1.5*tmp >= dv:
        step = 1
    elif 3.*tmp >= dv:
        step = 2
    elif 7.*tmp >= dv:
        step = 5
    else:
        step = 1
        factor = 0.1*factor

    return step, factor


def select_step(v1, v2, nv, hour=False):

    if v1 > v2:
        v1, v2 = v2, v1

    A = np.array

    dv = float(v2 - v1) / nv

    if hour:
        _select_step = select_step_hour
        cycle = 24.
    else:
        _select_step = select_step_degree
        cycle = 360.

    # for degree
    if dv > 1./3600.:
        #print "degree"
        step, factor = _select_step(dv)
    else:
        step, factor = select_step_sub(dv*3600.)
        #print "feac", step, factor

        factor = factor * 3600.


    f1, f2, fstep = v1*factor, v2*factor, step/factor
    levs = np.arange(math.floor(f1/step), math.ceil(f2/step)+0.5,
                     1, dtype="i") * step

    # n : number valid levels. If there is a cycle, e.g., [0, 90, 180,
    # 270, 360], the a grid line needs to be extend from 0 to 360, so
    # we need to return the whole array. However, the last level (360)
    # needs to be ignored often. In this case, so we return n=4.

    n = len(levs)


    # we need to check the range of values
    # for example, -90 to 90, 0 to 360,


    if factor == 1. and (levs[-1] >= levs[0]+cycle): # check for cycle
        nv = int(cycle / step)
        levs = np.arange(0, nv, 1) * step
        n = len(levs)

    return np.array(levs), n, factor


def select_step24(v1, v2, nv):
    v1, v2 = v1/15., v2/15.
    levs, n, factor =  select_step(v1, v2, nv, hour=True)
    return levs*15., n, factor

def select_step360(v1, v2, nv):
    return select_step(v1, v2, nv, hour=False)




class LocatorHMS(object):
    def __init__(self, den):
        self.den = den
    def __call__(self, v1, v2):
        return select_step24(v1, v2, self.den)


class LocatorDMS(object):
    def __init__(self, den):
        self.den = den
    def __call__(self, v1, v2):
        return select_step360(v1, v2, self.den)


class FormatterHMS(object):
    def __call__(self, direction, factor, values): # hour
        if len(values) == 0:
            return []
        #ss = [[-1, 1][v>0] for v in values]  #not py24 compliant
        values = np.asarray(values)
        ss = np.where(values>0, 1, -1)
        values = np.abs(values)/15.

        if factor == 1:
            return ["$%d^{\mathrm{h}}$" % (int(v),) for v in values]
        elif factor == 60:
            return ["$%d^{\mathrm{h}}\,%02d^{\mathrm{m}}$" % (s*floor(v/60.), v%60) \
                    for s, v in zip(ss, values)]
        elif factor == 3600:
            if ss[-1] == -1:
                inverse_order = True
                values = values[::-1]
            else:
                inverse_order = False
            degree = floor(values[0]/3600.)
            hm_fmt = "$%d^{\mathrm{h}}\,%02d^{\mathrm{m}}\,"
            s_fmt = "%02d^{\mathrm{s}}$"
            l_hm_old = ""
            r = []
            for v in values-3600*degree:
                l_hm = hm_fmt % (ss[0]*degree, floor(v/60.))
                l_s = s_fmt % (v%60,)
                if l_hm != l_hm_old:
                    l_hm_old = l_hm
                    l = l_hm + l_s
                else:
                    l = "$"+l_s
                r.append(l)
            if inverse_order:
                return r[::-1]
            else:
                return r
        #return [fmt % (ss[0]*degree, floor(v/60.), v%60) \
        #        for s, v in zip(ss, values-3600*degree)]
        else: # factor > 3600.
            return [r"$%s^{\mathrm{h}}$" % (str(v),) for v in ss*values]


class FormatterDMS(object):
    def __call__(self, direction, factor, values):
        if len(values) == 0:
            return []
        #ss = [[-1, 1][v>0] for v in values] #not py24 compliant
        values = np.asarray(values)
        ss = np.where(values>0, 1, -1)
        values = np.abs(values)
        if factor == 1:
            return ["$%d^{\circ}$" % (s*int(v),) for (s, v) in zip(ss, values)]
        elif factor == 60:
            return ["$%d^{\circ}\,%02d^{\prime}$" % (s*floor(v/60.), v%60) \
                    for s, v in zip(ss, values)]
        elif factor == 3600:
            if ss[-1] == -1:
                inverse_order = True
                values = values[::-1]
            else:
                inverse_order = False
            degree = floor(values[0]/3600.)
            hm_fmt = "$%d^{\circ}\,%02d^{\prime}\,"
            s_fmt = "%02d^{\prime\prime}$"
            l_hm_old = ""
            r = []
            for v in values-3600*degree:
                l_hm = hm_fmt % (ss[0]*degree, floor(v/60.))
                l_s = s_fmt % (v%60,)
                if l_hm != l_hm_old:
                    l_hm_old = l_hm
                    l = l_hm + l_s
                else:
                    l = "$"+l_s
                r.append(l)
            if inverse_order:
                return r[::-1]
            else:
                return r
            #return [fmt % (ss[0]*degree, floor(v/60.), v%60) \
            #        for s, v in zip(ss, values-3600*degree)]
        else: # factor > 3600.
            return [r"$%s^{\circ}$" % (str(v),) for v in ss*values]




class ExtremeFinderCycle(ExtremeFinderSimple):
    """
    When there is a cycle, e.g., longitude goes from 0-360.
    """
    def __init__(self,
                 nx, ny,
                 lon_cycle = 360.,
                 lat_cycle = None,
                 lon_minmax = None,
                 lat_minmax = (-90, 90)
                 ):
        #self.transfrom_xy = transform_xy
        #self.inv_transfrom_xy = inv_transform_xy
        self.nx, self.ny = nx, ny
        self.lon_cycle, self.lat_cycle = lon_cycle, lat_cycle
        self.lon_minmax = lon_minmax
        self.lat_minmax = lat_minmax


    def __call__(self, transform_xy, x1, y1, x2, y2):
        """
        get extreme values.

        x1, y1, x2, y2 in image coordinates (0-based)
        nx, ny : number of dvision in each axis
        """
        x_, y_ = np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny)
        x, y = np.meshgrid(x_, y_)
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        # iron out jumps, but algorithm should be improved.
        # Tis is just naive way of doing and my fail for some cases.
        if self.lon_cycle is not None:
            lon0 = np.nanmin(lon)
            lon -= 360. * ((lon - lon0) > 180.)
        if self.lat_cycle is not None:
            lat0 = np.nanmin(lat)
            lat -= 360. * ((lat - lat0) > 180.)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
                 self._adjust_extremes(lon_min, lon_max, lat_min, lat_max)

        return lon_min, lon_max, lat_min, lat_max


    def _adjust_extremes(self, lon_min, lon_max, lat_min, lat_max):

        lon_min, lon_max, lat_min, lat_max = \
                 self._add_pad(lon_min, lon_max, lat_min, lat_max)

        # check cycle
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        if self.lat_cycle:
            lat_max = min(lat_max, lat_min + self.lat_cycle)

        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)

        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)

        return lon_min, lon_max, lat_min, lat_max





if __name__ == "__main__":
    #test2()
    print select_step360(21.2, 33.3, 5)
    print select_step360(20+21.2/60., 21+33.3/60., 5)
    print select_step360(20.5+21.2/3600., 20.5+33.3/3600., 5)
    print select_step360(20+21.2/60., 20+53.3/60., 5)
