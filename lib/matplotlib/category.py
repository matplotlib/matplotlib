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
        """Uses axis.unit_data map to encode data as floats."""
        # We also need to pass numbers through.
        if np.issubdtype(np.asarray(value).dtype.type, np.number):
            return value
        else:
            axis.unit_data.update(value)
            return np.vectorize(axis.unit_data._mapping.__getitem__)(value)

    @staticmethod
    def axisinfo(unit, axis):
        # Note that mapping may get mutated by later calls to plotting methods,
        # so the locator and formatter must dynamically recompute locs and seq.
        return units.AxisInfo(
            majloc=StrCategoryLocator(axis.unit_data._mapping),
            majfmt=StrCategoryFormatter(axis.unit_data._mapping))

    @staticmethod
    def default_units(data, axis):
        # the conversion call stack is default_units->axis_info->convert
        if axis.unit_data is None:
            axis.unit_data = UnitData(data)
        else:
            axis.unit_data.update(data)
        return None


class StrCategoryLocator(ticker.FixedLocator):
    def __init__(self, mapping):
        self._mapping = mapping
        self.nbins = None

    @property
    def locs(self):
        return list(self._mapping.values())


class StrCategoryFormatter(ticker.FixedFormatter):
    def __init__(self, mapping):
        self._mapping = mapping
        self.offset_string = ""

    @property
    def seq(self):
        out = []
        for key in self._mapping:
            # So that we support bytes input.
            out.append(key.decode("latin-1") if isinstance(key, bytes)
                       else key)
        return out


class UnitData(object):
    def __init__(self, data):
        """Create mapping between unique categorical values and numerical id.

        Parameters
        ----------
        data: iterable
            sequence of values
        """
        self._mapping = {}
        self._counter = itertools.count()
        self.update(data)

    def update(self, data):
        if isinstance(data, six.string_types):
            data = [data]
        sorted_unique = OrderedDict.fromkeys(data)
        for s in sorted_unique:
            if s in self._mapping:
                continue
            self._mapping[s] = next(self._counter)


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
