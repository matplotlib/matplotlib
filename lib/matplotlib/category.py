"""Helpers for categorical data.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import collections
from collections import OrderedDict
from distutils.version import LooseVersion
import itertools

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker


if LooseVersion(np.__version__) >= LooseVersion('1.8.0'):
    def shim_array(data):
        return np.array(data, dtype=np.unicode)
else:
    def shim_array(data):
        if (isinstance(data, six.string_types) or
                not isinstance(data, collections.Iterable)):
            data = [data]
        try:
            data = [str(d) for d in data]
        except UnicodeEncodeError:
            # this yields gibberish but unicode text doesn't
            # render under numpy1.6 anyway
            data = [d.encode('utf-8', 'ignore').decode('utf-8')
                    for d in data]
        return np.array(data, dtype=np.unicode)


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode data as floats."""
        vmap = dict(zip(axis.unit_data.seq, axis.unit_data.locs))

        if isinstance(value, six.string_types):
            return vmap[value]

        vals = shim_array(value)

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
        """Create mapping between unique categorical values and numerical id.

        Parameters
        ----------
        data: iterable
            sequence of values
        """
        self.seq, self.locs = [], []
        self._counter = itertools.count()
        self.update(data)

    def update(self, data):
        data = np.atleast_1d(shim_array(data))
        sorted_unique = list(OrderedDict(zip(data, itertools.repeat(None))))
        for s in sorted_unique:
            if s in self.seq:
                continue
            self.seq.append(s)
            if s in UnitData.spdict:
                self.locs.append(UnitData.spdict[s])
            else:
                self.locs.append(next(self._counter))


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
