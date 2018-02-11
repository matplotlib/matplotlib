"""
catch all for categorical functions
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict
import itertools
from matplotlib import ticker, units
import six

import numpy as np


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Use axis.units mapping to map categorical data to floats."""
        def getter(k):
            if not isinstance(k, six.text_type):
                k = k.decode('utf-8')
            return axis.units._mapping[k]
        # We also need to pass numbers through.
        if np.issubdtype(np.asarray(value).dtype.type, np.number):
            return value
        else:
            axis.units.update(value)
            str2idx = np.vectorize(getter, otypes=[float])
            return str2idx(value)

    @staticmethod
    def axisinfo(unit, axis):
        majloc = StrCategoryLocator(axis.units)
        majfmt = StrCategoryFormatter(axis.units)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        return UnitData()


class StrCategoryLocator(ticker.Locator):
    def __init__(self, unit_data):
        self._unit_data = unit_data

    def __call__(self):
        return list(self._unit_data._mapping.values())

    def tick_values(self, vmin, vmax):
        return self()


class StrCategoryFormatter(ticker.Formatter):
    def __init__(self, unit_data):
        self._unit_data = unit_data

    def __call__(self, x, pos=None):
        if pos is None:
            return ""
        r_mapping = {v: k for k, v in self._unit_data._mapping.items()}
        return r_mapping.get(int(x), '')


class UnitData(object):
    valid_types = tuple(set(six.string_types +
                            (bytes, six.text_type, np.str_, np.bytes_)))

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
        self._mapping = OrderedDict()
        for k, v in OrderedDict(data).items():
            if not isinstance(k, self.valid_types):
                raise TypeError("{val!r} is not a string".format(val=k))
            if not isinstance(k, six.text_type):
                k = k.decode('utf-8')
            self._mapping[k] = int(v)
        if self._mapping:
            start = max(self._mapping.values()) + 1
        else:
            start = 0
        self._counter = itertools.count(start=start)

    def update(self, data):
        if isinstance(data, self.valid_types):
            data = [data]
        sorted_unique = OrderedDict.fromkeys(data)
        for val in sorted_unique:
            if not isinstance(val, self.valid_types):
                raise TypeError("{val!r} is not a string".format(val=val))
            if not isinstance(val, six.text_type):
                val = val.decode('utf-8')
            if val in self._mapping:
                continue
            self._vals.append(val)
            self._mapping[val] = next(self._counter)


# Connects the convertor to matplotlib

units.registry[str] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
