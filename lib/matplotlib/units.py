"""
The classes here provide support for using custom classes with
matplotlib, eg those that do not expose the array interface but know
how to converter themselves to arrays.  It also supoprts classes with
units and units conversion.  Use cases include converters for custom
objects, eg a list of datetime objects, as well as for objects that
are unit aware.  We don't assume any particular units implementation,
rather a units implementation must provide a ConversionInterface, and
the register with the Registry converter dictionary.  For example,
here is a complete implementation which support plotting with native
datetime objects


    import matplotlib.units as units
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker
    import datetime

    class DateConverter(units.ConversionInterface):

        def convert(value, unit):
            'convert value to a scalar or array'
            return dates.date2num(value)
        convert = staticmethod(convert)

        def axisinfo(unit):
            'return major and minor tick locators and formatters'
            if unit!='date': return None
            majloc = dates.AutoDateLocator()
            majfmt = dates.AutoDateFormatter(majloc)
            return AxisInfo(majloc=majloc,
                            majfmt=majfmt,
                            label='date')
        axisinfo = staticmethod(axisinfo)


        def default_units(x):
            'return the default unit for x or None'
            return 'date'
        default_units = staticmethod(default_units)

    # finally we register our object type with a converter
    units.registry[datetime.date] = DateConverter()

"""
import numpy as np
from matplotlib.cbook import iterable, is_numlike

class AxisInfo:
    'information to support default axis labeling and tick labeling'
    def __init__(self, majloc=None, minloc=None,
                 majfmt=None, minfmt=None, label=None):
        """
        majloc and minloc: TickLocators for the major and minor ticks
        majfmt and minfmt: TickFormatters for the major and minor ticks
        label: the default axis label

        If any of the above are None, the axis will simply use the default
        """
        self.majloc = majloc
        self.minloc = minloc
        self.majfmt = majfmt
        self.minfmt = minfmt
        self.label = label


class ConversionInterface:
    """
    The minimal interface for a converter to take custom instances (or
    sequences) and convert them to values mpl can use
    """
    def axisinfo(unit):
        'return an units.AxisInfo instance for unit'
        return None
    axisinfo = staticmethod(axisinfo)

    def default_units(x):
        'return the default unit for x or None'
        return None
    default_units = staticmethod(default_units)

    def convert(obj, unit):
        """
        convert obj using unit.  If obj is a sequence, return the
        converted sequence.  The ouput must be a sequence of scalars
        that can be used by the numpy array layer
        """
        return obj
    convert = staticmethod(convert)

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
    is_numlike = staticmethod(is_numlike)

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
            # if this is anything but an object array, we'll assume
            # there are no custom units
            if isinstance(x, np.ndarray) and x.dtype != np.object:
                return None

            for thisx in x:
                converter = self.get_converter( thisx )
                return converter

        #DISABLED self._cached[idx] = converter
        return converter


registry = Registry()
