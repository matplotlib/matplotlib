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


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode
        data as floats
        """
        vmap = dict(zip(axis.unit_data.seq, axis.unit_data.locs))

        if isinstance(value, six.string_types):
            return vmap[value]

        vals = np.array(value, dtype=np.unicode)
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
        strdata = np.array(data, dtype=np.unicode)
        # np.unique makes dateframes work
        new_s = [d for d in np.unique(strdata) if d not in self.seq]
        for ns in new_s:
            self.seq.append(ns)
            if ns in UnitData.spdict:
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
