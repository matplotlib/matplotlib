"""
catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """Uses axis.unit_data map to encode
        data as integers
        """

        if isinstance(value, six.string_types):
            return dict(axis.unit_data)[value]

        vals = np.asarray(value, dtype='str')
        for label, loc in axis.unit_data:
            vals[vals == label] = loc
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


def map_categories(data, old_map=[], sort=True):
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
    spdict = {'nan': -1, 'inf': -2, '-inf': -3}

    # cast all data to str
    strdata = [str(d) for d in data]

    uniq = set(strdata)

    category_map = old_map.copy()

    if old_map:
        olabs, okeys = zip(*old_map)
        olabs, okeys = set(olabs), set(okeys)
        svalue = max(okeys) + 1
    else:
        olabs, okeys = set(), set()
        svalue = 0

    new_labs = (uniq - olabs)

    missing = (new_labs & set(spdict.keys()))
    category_map.extend([(m, spdict[m]) for m in missing])

    new_labs = (new_labs - missing)
    if sort:
        new_labs = list(new_labs)
        new_labs.sort()

    new_locs = range(svalue, svalue + len(new_labs))
    category_map.extend(list(zip(new_labs, new_locs)))
    return category_map


class StrCategoryLocator(ticker.FixedLocator):
    def __init__(self, locs):
        super(StrCategoryLocator, self).__init__(locs, None)


class StrCategoryFormatter(ticker.FixedFormatter):
    def __init__(self, seq):
        super(StrCategoryFormatter, self).__init__(seq)


# Connects the convertor to matplotlib
units.registry[bytearray] = StrCategoryConverter()
units.registry[str] = StrCategoryConverter()

if six.PY3:
    units.registry[bytes] = StrCategoryConverter()
elif six.PY2:
    units.registry[unicode] = StrCategoryConverter()
