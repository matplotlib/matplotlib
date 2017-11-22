# -*- coding: utf-8 OA -*-

"""
catch all for categorical functions
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable, Sequence, OrderedDict
import itertools
import numbers
from matplotlib import cbook, ticker, units
import six

from collections import OrderedDict
import itertools

import numpy as np


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Use axis.units mapping tncode data as floats."""

        # We also need to pass numbers through.
        if np.issubdtype(np.asarray(value).dtype.type, np.number):
            return value
        else:
            axis.units.update(value)
            str2idx = np.vectorize(axis.units._mapping.__getitem__,
                                   otypes=[float])
            return str2idx(value)

    @staticmethod
    def axisinfo(unit, axis):
        majloc = StrCategoryLocator(axis.unit_data._locs)
        majfmt = StrCategoryFormatter(axis.unit_data._seq)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        return UnitData()



class StrCategoryLocator(ticker.FixedLocator):
    def __init__(self, locs):
        self.locs = locs
        self.nbins = None


class StrCategoryFormatter(ticker.FixedFormatter):
    def __init__(self, seq):
        self.seq = seq
        self.offset_string = ''


class UnitData(object):
    def __init__(self, data=None):
        """Create mapping between unique categorical values and numerical id.

        Parameters
        ----------
        data : Mapping[str, int]
            The initial categories.  May be `None`.

        """
        self._vals = []
        if data is None:
            data = ()
        self._mapping = OrderedDict(data)
        for k, v in self._mapping.items():
            if not isinstance(k, six.text_type):
                raise TypeError("{val!r} is not a string".format(val=k))
            self._mapping[k] = int(v)
        if self._mapping:
            start = max(self._mapping.values()) + 1
        else:
            start = 0
        self._counter = itertools.count(start=start)

    def update(self, data):
        if isinstance(data, six.string_types):
            data = [data]
        sorted_unique = OrderedDict.fromkeys(data)
        for val in sorted_unique:
            if val in self._mapping:
                continue
            if not isinstance(val, six.text_type):
                raise TypeError("{val!r} is not a string".format(val))
            self._vals.append(val)
            self._mapping[val] = next(self._counter)


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
