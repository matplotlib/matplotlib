# -*- coding: utf-8 OA-*-za
"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np

import matplotlib.cbook as cbook
import matplotlib.units as units
import matplotlib.ticker as ticker


#  pure hack for numpy 1.6 support
from distutils.version import LooseVersion

NP_NEW = (LooseVersion(np.version.version) >= LooseVersion('1.7'))


def to_array(data, maxlen=100):
    if NP_NEW:
        return np.array(data, dtype=np.unicode)
    if cbook.is_scalar_or_string(data):
        data = [data]
    try:
        vals = np.array(data, dtype=('|S', maxlen))
    except UnicodeEncodeError:
        # this yields gibberish
        vals = np.array([convert_to_string(d) for d in data])
    return vals


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode
        data as floats
        """
        vmap = dict(zip(axis.unit_data.seq, axis.unit_data.locs))

        if isinstance(value, six.string_types):
            return vmap[value]

        vals = to_array(value)
        for lab, loc in vmap.items():
            vals[vals == lab] = loc

        return vals.astype('float')

    @staticmethod
    def axisinfo(unit, axis):
        majloc = StrCategoryLocator(axis.unit_data.locs)
        majfmt = StrCategoryFormatter(axis.unit_data.seq)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        # the conversion call stack is:
        # default_units->axis_info->convert
        if axis.unit_data is None:
            axis.unit_data = UnitData(data)
        else:
            axis.unit_data.update(data)
        return None


class StrCategoryLocator(ticker.FixedLocator):
    def __init__(self, locs):
        self.locs = locs
        self.nbins = None


class StrCategoryFormatter(ticker.FixedFormatter):
    def __init__(self, seq):
        self.seq = seq
        self.offset_string = ''


def convert_to_string(value):
    """Helper function for numpy 1.6, can be replaced with
    np.array(...,dtype=unicode) for all later versions of numpy"""

    if isinstance(value, six.string_types):
        pass
    elif np.isfinite(value):
        value = np.asarray(value, dtype=str)[np.newaxis][0]
    elif np.isnan(value):
        value = 'nan'
    elif np.isposinf(value):
        value = 'inf'
    elif np.isneginf(value):
        value = '-inf'
    else:
        raise ValueError("Unconvertable {}".format(value))
    return value


class UnitData(object):
    # debatable makes sense to special code missing values
    spdict = {'nan': -1.0, 'inf': -2.0, '-inf': -3.0}

    def __init__(self, data):
        """Create mapping between unique categorical values
        and numerical identifier
        Paramters
        ---------
        data: iterable
            sequence of values
        """
        self.seq, self.locs = [], []
        self._set_seq_locs(data, 0)

    def update(self, new_data):
        # so as not to conflict with spdict
        value = max(max(self.locs) + 1, 0)
        self._set_seq_locs(new_data, value)

    def _set_seq_locs(self, data, value):
        # magic to make it work under np1.6
        strdata = to_array(data)
        # np.unique makes dateframes work
        new_s = [d for d in np.unique(strdata) if d not in self.seq]
        for ns in new_s:
            self.seq.append(convert_to_string(ns))
            if ns in UnitData.spdict.keys():
                self.locs.append(UnitData.spdict[ns])
            else:
                self.locs.append(value)
                value += 1


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
