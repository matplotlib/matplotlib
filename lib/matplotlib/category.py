# -*- coding: utf-8 -*-
"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable, OrderedDict
import itertools

import six

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker

# np 1.6/1.7 support
from distutils.version import LooseVersion

VALID_TYPES = tuple(set(six.string_types +
                        (bytes, six.text_type, np.str_, np.bytes_)))


def to_str(value):
    """Helper function to turn values to strings.
    """
    # Note: This function is only used by StrCategoryFormatter
    if LooseVersion(np.__version__) < LooseVersion('1.7.0'):
        if (isinstance(value, (six.text_type, np.unicode))):
            value = value.encode('utf-8', 'ignore').decode('utf-8')
    if isinstance(value, (np.bytes_, six.binary_type)):
        value = value.decode(encoding='utf-8')
    elif not isinstance(value, (np.str_, six.string_types)):
        value = str(value)
    return value


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.units to encode string data as floats

        Parameters
        ----------
        value: string, iterable
            value or list of values to plot
        unit:
        axis:
        """
        # dtype = object preserves numerical pass throughs
        values = np.atleast_1d(np.array(value, dtype=object))

        # pass through sequence of non binary numbers
        if all((units.ConversionInterface.is_numlike(v) and
                not isinstance(v, VALID_TYPES)) for v in values):
            return np.asarray(values, dtype=float)

        # force an update so it also does type checking
        axis.units.update(values)

        str2idx = np.vectorize(axis.units._mapping.__getitem__,
                               otypes=[float])

        mapped_value = str2idx(values)
        return mapped_value

    @staticmethod
    def axisinfo(unit, axis):
        """Sets the axis ticks and labels
        """
        # locator and formatter take mapping dict because
        # args need to be pass by reference for updates
        majloc = StrCategoryLocator(axis.units)
        majfmt = StrCategoryFormatter(axis.units)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data=None, axis=None):
        # the conversion call stack is supposed to be
        # default_units->axis_info->convert
        if axis.units is None:
            axis.set_units(UnitData(data))
        else:
            axis.units.update(data)
        return axis.units


class StrCategoryLocator(ticker.Locator):
    """tick at every integer mapping of the string data"""
    def __init__(self, units):
        """
        Parameters
        -----------
        units: dict
              (string, integer) mapping
        """
        self._units = units

    def __call__(self):
        return list(self._units._mapping.values())

    def tick_values(self, vmin, vmax):
        return self()


class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick"""
    def __init__(self, units):
        """
        Parameters
        ----------
        units: dict
              (string, integer) mapping
        """
        self._units = units

    def __call__(self, x, pos=None):
        if pos is None:
            return ""
        r_mapping = {v: to_str(k) for k, v in self._units._mapping.items()}
        return r_mapping.get(int(np.round(x)), '')


class UnitData(object):
    def __init__(self, data=None):
        """Create mapping between unique categorical values
        and integer identifiers
        ----------
        data: iterable
              sequence of string values
        """
        if data is None:
            data = ()
        self._mapping = OrderedDict()
        self._counter = itertools.count(start=0)
        self.update(data)

    def update(self, data):
        """Maps new values to integer identifiers.

        Paramters
        ---------
        data: iterable
              sequence of string values

        Raises
        ------
        TypeError
              If the value in data is not a string, unicode, bytes type
        """

        if (isinstance(data, VALID_TYPES) or
                not isinstance(data, Iterable)):
            data = [data]

        unsorted_unique = OrderedDict.fromkeys(data)
        for val in unsorted_unique:
            if not isinstance(val, VALID_TYPES):
                raise TypeError("{val!r} is not a string".format(val=val))
            if val not in self._mapping:
                self._mapping[val] = next(self._counter)


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[np.str_] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[np.bytes_] = StrCategoryConverter()
