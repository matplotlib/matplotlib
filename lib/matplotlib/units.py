"""
The classes here provide support for using custom classes with
matplotlib, eg those that do not expose the array interface but know
how to converter themselves to arrays.  It also supoprts classes with
units and units conversion.  Use cases include converters for custom
objects, eg a list of datetime objects, as well as for objects that
are unit aware.  We don't assume any particular units implementation,
rather a units implementation must provide a ConversionInterface, and
the register with the Registry converter dictionary.  For example,
here is a complete implementation which supports plotting with native
datetime objects::


    import matplotlib.units as units
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker
    import datetime

    class DateConverter(units.ConversionInterface):

        @staticmethod
        def convert(value, unit, axis):
            'convert value to a scalar or array'
            return dates.date2num(value)

        @staticmethod
        def axisinfo(unit, axis):
            'return major and minor tick locators and formatters'
            if unit!='date': return None
            majloc = dates.AutoDateLocator()
            majfmt = dates.AutoDateFormatter(majloc)
            return AxisInfo(majloc=majloc,
                            majfmt=majfmt,
                            label='date')

        @staticmethod
        def default_units(x, axis):
            'return the default unit for x or None'
            return 'date'

    # finally we register our object type with a converter
    units.registry[datetime.date] = DateConverter()

"""
import numpy as np
from matplotlib.cbook import iterable, is_numlike, is_string_like

class AxisInfo:
    'information to support default axis labeling and tick labeling, and default limits'
    def __init__(self, majloc=None, minloc=None,
                 majfmt=None, minfmt=None, label=None,
                 default_limits=None):
        """
        majloc and minloc: TickLocators for the major and minor ticks
        majfmt and minfmt: TickFormatters for the major and minor ticks
        label: the default axis label
        default_limits: the default min, max of the axis if no data is present
        If any of the above are None, the axis will simply use the default
        """
        self.majloc = majloc
        self.minloc = minloc
        self.majfmt = majfmt
        self.minfmt = minfmt
        self.label = label
        self.default_limits = default_limits


class ConversionInterface:
    """
    The minimal interface for a converter to take custom instances (or
    sequences) and convert them to values mpl can use
    """
    @staticmethod
    def axisinfo(unit, axis):
        'return an units.AxisInfo instance for axis with the specified units'
        return None

    @staticmethod
    def default_units(x, axis):
        'return the default unit for x or None for the given axis'
        return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        convert obj using unit for the specified axis.  If obj is a sequence,
        return the converted sequence.  The ouput must be a sequence of scalars
        that can be used by the numpy array layer
        """
        return obj

    @staticmethod
    def is_numlike(x):
        """
        The matplotlib datalim, autoscaling, locators etc work with
        scalars which are the units converted to floats given the
        current unit.  The converter may be passed these floats, or
        arrays of them, even when units are set.  Derived conversion
        interfaces may opt to pass plain-ol unitless numbers through
        the conversion interface and this is a helper function for
        them.
        """
        if iterable(x):
            for thisx in x:
                return is_numlike(thisx)
        else:
            return is_numlike(x)

class Registry(dict):
    """
    register types with conversion interface
    """
    def __init__(self):
        dict.__init__(self)
        self._cached = {}

    def get_converter(self, x):
        'get the converter interface instance for x, or None'

        if not len(self): return None # nothing registered
        #DISABLED idx = id(x)
        #DISABLED cached = self._cached.get(idx)
        #DISABLED if cached is not None: return cached

        converter = None
        classx = getattr(x, '__class__', None)

        if classx is not None:
            converter = self.get(classx)

        if converter is None and iterable(x):
            for thisx in x:
                # Make sure that recursing might actually lead to a solution, if
                # we are just going to re-examine another item of the same kind,
                # then do not look at it.
                if classx and classx != getattr(thisx, '__class__', None):
                    converter = self.get_converter( thisx )
                    return converter

        #DISABLED self._cached[idx] = converter
        return converter


registry = Registry()
