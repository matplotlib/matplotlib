# -*- coding: utf-8 OA-*-za
"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable, Sequence, OrderedDict
import itertools
import numbers

import six

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker

# np 1.6/1.7 support
from distutils.version import LooseVersion


def to_str(value):
    if LooseVersion(np.__version__) < LooseVersion('1.7.0'):
        if (isinstance(value, (six.text_type, np.unicode))):
            value = value.encode('utf-8', 'ignore').decode('utf-8')
    if isinstance(value, (bytes, np.bytes_, six.binary_type)):
        value = value.decode(encoding='utf-8')
    elif isinstance(value, (bytes, np.bytes_, six.binary_type)):
        return value.decode(encoding='utf-8')
    elif not isinstance(value, (str, np.str_, six.text_type)):
        value = str(value)
    return value


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode
        data as floats
        """
        if isinstance(value, six.string_types):
            return axis.unit_data._mapping[value]

        # dtype=object preserves 42, '42' distinction on scatter
        values = np.atleast_1d(np.array(value, dtype=object))
        if units.ConversionInterface.is_numlike(value):
            return np.array([axis.unit_data._mapping.get(v, v)
                             for v in values])

        if hasattr(axis.unit_data, 'update'):
            axis.unit_data.update(values)

        str2idx = np.vectorize(axis.unit_data._mapping.__getitem__,
                               otypes=[float])

        mapped_value = str2idx(values)
        return mapped_value

    @staticmethod
    def axisinfo(unit, axis):
        majloc = StrCategoryLocator(axis.unit_data._locs)
        majfmt = StrCategoryFormatter(axis.unit_data._seq)
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
    def __init__(self, data):
        """Create mapping between unique categorical values
        and numerical identifier

        Parameters
        ----------
        data: iterable
              sequence of values
        """
        # seq, loc need to be pass by reference or there needs to be
        # a callback from Locator/Formatter on update
        self._seq, self._locs = [], []
        self._mapping = OrderedDict()
        self._counter = itertools.count()
        self.update(data)

    def _update_mapping(self, value):
        if value in self._mapping:
            return
        if isinstance(value, (float, complex)) and np.isnan(value):
            self._mapping[value] = np.nan
        else:
            self._mapping[value] = next(self._counter)
        self._seq.append(to_str(value))
        self._locs.append(self._mapping[value])
        return

    def update(self, data):
        if (isinstance(data, six.string_types) or
                not isinstance(data, Iterable)):
            self._update_mapping(data)
        else:
            unsorted_unique = OrderedDict.fromkeys(data)
            for ns in unsorted_unique:
                self._update_mapping(ns)

# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
