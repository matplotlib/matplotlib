"""Helpers for categorical data.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from collections import OrderedDict
import itertools

import numpy as np

from matplotlib import cbook, ticker, units


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Encode data as floats."""
        # We also need to pass numbers through.
        if np.issubdtype(np.asarray(value).dtype.type, np.number):
            return value
        else:
            unit.update(value)
            return np.vectorize(unit._val_to_idx.__getitem__)(value)

    @staticmethod
    def axisinfo(unit, axis):
        # Note that mapping may get mutated by later calls to plotting methods,
        # so the locator and formatter must dynamically recompute locs and seq.
        return units.AxisInfo(
            majloc=StrCategoryLocator(unit),
            majfmt=StrCategoryFormatter(unit))

    @staticmethod
    def default_units(data, axis):
        return _CategoricalUnit()


class StrCategoryLocator(ticker.Locator):
    def __init__(self, unit_data):
        self._unit_data = unit_data

    def __call__(self):
        return list(self._unit_data._val_to_idx.values())


class StrCategoryFormatter(ticker.Formatter):
    def __init__(self, unit_data):
        self._unit_data = unit_data

    def __call__(self, x, pos=None):
        if pos in range(len(self._unit_data._vals)):
            s = self._unit_data._vals[pos]
            if isinstance(s, bytes):
                s = s.decode("latin-1")
            return s
        else:
            return ""


class _CategoricalUnit(object):
    def __init__(self):
        """Create mapping between unique categorical values and numerical id.
        """
        self._vals = []
        self._val_to_idx = OrderedDict()
        self._counter = itertools.count()

    def update(self, data):
        if isinstance(data, six.string_types):
            data = [data]
        sorted_unique = OrderedDict.fromkeys(data)
        for val in sorted_unique:
            if val in self._val_to_idx:
                continue
            if not isinstance(val, (six.text_type, six.binary_type)):
                raise TypeError("Not a string")
            self._vals.append(val)
            self._val_to_idx[val] = next(self._counter)


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
