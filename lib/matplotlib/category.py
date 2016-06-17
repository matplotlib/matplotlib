# -*- coding: utf-8 OA-*-za
"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker


#  pure hack for numpy 1.6 support
from distutils.version import LooseVersion

NP_NEW = (LooseVersion(np.version.version) >= LooseVersion('1.7'))


def to_array(data, maxlen=100):
    if NP_NEW:
        return np.array(data, dtype=np.unicode)
    try:
        vals = np.array(data, dtype=('|S', maxlen))
    except UnicodeEncodeError:
        # pure hack
        vals = np.array([convert_to_string(d) for d in data])
    return vals


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode
        data as floats
        """
        vmap = dict(axis.unit_data)

        if isinstance(value, six.string_types):
            return vmap[value]

        vals = to_array(value)
        for lab, loc in axis.unit_data:
            vals[vals == lab] = loc

        return vals.astype('float')

    @staticmethod
    def axisinfo(unit, axis):
        seq, locs = zip(*axis.unit_data)
        majloc = StrCategoryLocator(locs)
        majfmt = StrCategoryFormatter(seq)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        # the conversion call stack is:
        # default_units->axis_info->convert
        axis.unit_data = map_categories(data, axis.unit_data)
        return None


class StrCategoryLocator(ticker.FixedLocator):
    def __init__(self, locs):
        super(StrCategoryLocator, self).__init__(locs, None)


class StrCategoryFormatter(ticker.FixedFormatter):
    def __init__(self, seq):
        super(StrCategoryFormatter, self).__init__(seq)


def convert_to_string(value):
    """Helper function for numpy 1.6, can be replaced with
    np.array(...,dtype=unicode) for all later versions of numpy"""

    if isinstance(value, six.string_types):
        return value
    if np.isfinite(value):
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


def map_categories(data, old_map=None):
    """Create mapping between unique categorical
    values and numerical identifier.

    Paramters
    ---------
    data: iterable
        sequence of values
    old_map: list of tuple, optional
        if not `None`, than old_mapping will be updated with new values and
        previous mappings will remain unchanged)
    sort: bool, optional
        sort keys by ASCII value

    Returns
    -------
    list of tuple
        [(label, ticklocation),...]

    """

    # code typical missing data in the negative range because
    # everything else will always have positive encoding
    # question able if it even makes sense
    spdict = {'nan': -1.0, 'inf': -2.0, '-inf': -3.0}

    if isinstance(data, six.string_types):
        data = [data]

    # will update this post cbook/dict support
    strdata = to_array(data)
    uniq = np.unique(strdata)

    if old_map:
        olabs, okeys = zip(*old_map)
        svalue = max(okeys) + 1
    else:
        old_map, olabs, okeys = [], [], []
        svalue = 0

    category_map = old_map[:]

    new_labs = [u for u in uniq if u not in olabs]
    missing = [nl for nl in new_labs if nl in spdict.keys()]

    category_map.extend([(m, spdict[m]) for m in missing])

    new_labs = [nl for nl in new_labs if nl not in missing]

    new_locs = np.arange(svalue, svalue + len(new_labs), dtype='float')
    category_map.extend(list(zip(new_labs, new_locs)))
    return category_map


# Connects the convertor to matplotlib
units.registry[str] = StrCategoryConverter()
units.registry[bytes] = StrCategoryConverter()
units.registry[six.text_type] = StrCategoryConverter()
