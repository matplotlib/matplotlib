# -*- coding: utf-8 OA-*-za
"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.cbook as cbook
import matplotlib.units as munits
import matplotlib.ticker as mticker

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


class StrCategoryConverter(munits.ConversionInterface):
    """Converts categorical (or string) data to numerical values

    Conversion typically happens in the following order:
    1. default_units:
        create unit_data category-integer mapping and binds to axis
    2. axis_info:
        set ticks/locator and labels/formatter
    3. convert:
        map input category data to integers using unit_data
    """
    @staticmethod
    def convert(value, unit, axis):
        """
        Encode value  as floats using axis.unit_data
        """
        vmap = dict(zip(axis.unit_data.seq, axis.unit_data.locs))

        if isinstance(value, six.string_types):
            return vmap.get(value, None)

        vals = to_array(value)
        for lab, loc in vmap.items():
            vals[vals == lab] = loc

        return vals.astype('float64')

    @staticmethod
    def axisinfo(unit, axis):
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is None
        *axis.unit_data* is used to set ticks and labels
        """
        majloc = StrCategoryLocator(axis.unit_data.locs)
        majfmt = StrCategoryFormatter(axis.unit_data.seq)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis, sort=True, normed=False):
        """
        Create mapping between string categories in *data*
        and integers, and store in *axis.unit_data*
        """
        if axis and axis.unit_data:
            axis.unit_data.update(data, sort)
            return axis.unit_data

        unit_data = UnitData(data, sort)
        if axis:
            axis.unit_data = unit_data
        return unit_data


class StrCategoryLocator(mticker.FixedLocator):
    """
    Ensures that every category has a tick by subclassing
    :class:`~matplotlib.ticker.FixedLocator`
    """
    def __init__(self, locs):
        self.locs = locs
        self.nbins = None


class StrCategoryFormatter(mticker.FixedFormatter):
    """
    Labels every category by subclassing
    :class:`~matplotlib.ticker.FixedFormatter`
    """
    def __init__(self, seq):
        self.seq = seq
        self.offset_string = ''


class CategoryNorm(mcolors.Normalize):
    """
    Preserves ordering of discrete values
    """
    def __init__(self, data):
        """
        *categories*
            distinct values for mapping

        Out-of-range values are mapped to np.nan
        """

        self.units = StrCategoryConverter()
        self.unit_data = None
        self.units.default_units(data,
                                 self, sort=False)
        self.loc2seq = dict(zip(self.unit_data.locs, self.unit_data.seq))
        self.vmin = min(self.unit_data.locs)
        self.vmax = max(self.unit_data.locs)

    def __call__(self, value, clip=None):
        # gonna have to go into imshow and undo casting
        value = np.asarray(value, dtype=np.int)
        ret = self.units.convert(value, None, self)
        # knock out values not in the norm
        mask = np.in1d(ret, self.unit_data.locs).reshape(ret.shape)
        # normalize ret & locs
        ret /= self.vmax
        return np.ma.array(ret, mask=~mask)

    def inverse(self, value):
        if not cbook.iterable(value):
            value = np.asarray(value)
        vscaled = np.asarray(value) * self.vmax
        return [self.loc2seq[int(vs)] for vs in vscaled]


def colors_from_categories(codings):
    """
    Helper routine to generate a cmap and a norm from a list
    of (color, value) pairs

    Parameters
    ----------
    codings : sequence of (key, value) pairs

    Returns
    -------
    (cmap, norm) : tuple containing a :class:`Colormap` and a \
                   :class:`Normalize` instance
    """
    if isinstance(codings, dict):
        codings = cbook.sanitize_sequence(codings.items())
    values, colors = zip(*codings)
    cmap = mcolors.ListedColormap(list(colors))
    norm = CategoryNorm(list(values))
    return cmap, norm


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
    # debatable if it makes sense to special code missing values
    spdict = {'nan': -1.0, 'inf': -2.0, '-inf': -3.0}

    def __init__(self, data, sort=True):
        """Create mapping between unique categorical values
        and numerical identifier
        Paramters
        ---------
        data: iterable
            sequence of values
        sort: bool
            sort input data, default is True
            False preserves input order
        """
        self.seq, self.locs = [], []
        self._set_seq_locs(data, 0, sort)
        self.sort = sort

    def update(self, new_data, sort=True):
        if sort:
            self.sort = sort
        # so as not to conflict with spdict
        value = max(max(self.locs) + 1, 0)
        self._set_seq_locs(new_data, value, self.sort)

    def _set_seq_locs(self, data, value, sort):
        # magic to make it work under np1.6
        strdata = to_array(data)

        # np.unique makes dateframes work
        if sort:
            unq = np.unique(strdata)
        else:
            _, idx = np.unique(strdata, return_index=~sort)
            unq = strdata[np.sort(idx)]

        new_s = [d for d in unq if d not in self.seq]
        for ns in new_s:
            self.seq.append(convert_to_string(ns))
            if ns in UnitData.spdict.keys():
                self.locs.append(UnitData.spdict[ns])
            else:
                self.locs.append(value)
                value += 1

# Connects the convertor to matplotlib
munits.registry[str] = StrCategoryConverter()
munits.registry[bytes] = StrCategoryConverter()
munits.registry[six.text_type] = StrCategoryConverter()
