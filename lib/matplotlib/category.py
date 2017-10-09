"""Helpers for categorical data.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from collections import OrderedDict
import itertools

import numpy as np

from matplotlib import units, ticker


def _to_str(s):
    return s.decode("ascii") if isinstance(s, bytes) else str(s)


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode data as floats."""
        mapping = axis.unit_data._mapping
        return (mapping[_to_str(value)] if np.isscalar(value)
                else np.array([mapping[_to_str(v)] for v in value], float))

    @staticmethod
    def axisinfo(unit, axis):
        # Note that mapping may get mutated by later calls to plotting methods,
        # so the locator and formatter must dynamically recompute locs and seq.
        majloc = StrCategoryLocator(axis.unit_data._mapping)
        majfmt = StrCategoryFormatter(axis.unit_data._mapping)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

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
        return list(self._mapping)


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
        sorted_unique = OrderedDict.fromkeys(map(_to_str, data))
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
