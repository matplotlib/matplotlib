from datetime import timedelta

import numpy as np

from matplotlib import _api, units, ticker

VALID_UNITS = ['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks']
NUMPY_CODES = {'days': 'D',
               'seconds': 's',
               'microseconds': 'us',
               'milliseconds': 'ms',
               'minutes': 'm',
               'hours': 'h',
               'weeks': 'W'}

class TimeDeltaConverter(units.ConversionInterface):
    f"""
    Interface for converting timedeltas.

    Both builtin Python `datetime.timedelta` objects and `numpy.timedelta64`
    objects are supported.

    Notes
    -----
    This interface supports converting to one of the builtin Python time
    intervals: {VALID_UNITS}. Although `numpy.timedelta64` supports smaller
    time intervals than microseconds, these are not yet supported.
    """
    @staticmethod
    def _validate_unit(unit):
        _api.check_in_list(VALID_UNITS, unit=unit)

    @staticmethod
    def _is_np_timedelta(value):
        return ((isinstance(value, np.ndarray) and
                 np.issubdtype(value.dtype, np.timedelta64)) or
               isinstance(value, np.timedelta64))

    @staticmethod
    def convert(value, unit, axis):
        f"""
        Convert a timedelta value to a scalar or array.

        Parameters
        ----------
        value : datetime.timedelta, list[datetime.timedelta], numpy.timedelta64
            Value to convert.
        unit : str
            One of {VALID_UNITS}.
        axis : matplotlib.axis.Axis
            Not used.
        """
        TimeDeltaConverter._validate_unit(unit)

        if isinstance(value, list):
            return [TimeDeltaConverter.convert(v, unit, axis) for v in value]
        elif TimeDeltaConverter._is_np_timedelta(value):
            return value / np.timedelta64(1, NUMPY_CODES[unit])
        else:
            unit_delta = timedelta(**{unit: 1})
            return value / unit_delta

    @staticmethod
    def axisinfo(unit, axis):
        "Return major and minor tick locators and formatters."
        TimeDeltaConverter._validate_unit(unit)

        majloc = ticker.AutoLocator()
        majfmt = ticker.ScalarFormatter()
        return units.AxisInfo(majloc=majloc,
                              majfmt=majfmt,
                              label=unit.capitalize())

    @staticmethod
    def default_units(value, axis):
        return 'seconds'


units.registry[timedelta] = TimeDeltaConverter()
units.registry[np.timedelta64] = TimeDeltaConverter()
