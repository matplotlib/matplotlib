"""EpochConverter module containing class EpochConverter."""

from matplotlib import cbook
import matplotlib.units as units
import matplotlib.dates as date_ticker

__all__ = ['EpochConverter']


class EpochConverter(units.ConversionInterface):
    """
    Provides Matplotlib conversion functionality for Monte Epoch and Duration
    classes.
    """

    # julian date reference for "Jan 1, 0001" minus 1 day because
    # Matplotlib really wants "Jan 0, 0001"
    jdRef = 1721425.5 - 1
    day_in_seconds = 86400.0

    @classmethod
    def axisinfo(cls, unit, axis):
        """: Returns information on how to handle an axis that has Epoch data.

        = INPUT VARIABLES
        - unit     The units to use for a axis with Epoch data.

        = RETURN VALUE
        - Returns a AxisInfo data structure that contains
          minor/major formatters, major/minor locators, and default
          label information.
        """

        majloc = date_ticker.AutoDateLocator()
        majfmt = date_ticker.AutoDateFormatter(majloc)

        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label=unit)

    @classmethod
    def float2epoch(cls, value, unit):
        """: Convert a Matplotlib floating-point date into an Epoch of the
              specified units.

        = INPUT VARIABLES
        - value     The Matplotlib floating-point date.
        - unit      The unit system to use for the Epoch.

        = RETURN VALUE
        - Returns the value converted to an Epoch in the specified time system.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        secPastRef = value * cls.day_in_seconds * U.UnitDbl(1.0, 'sec')
        return U.Epoch(unit, secPastRef, cls.jdRef)

    @classmethod
    def epoch2float(cls, value, unit):
        """: Convert an Epoch value to a float suitable for plotting as a
              python datetime object.

        = INPUT VARIABLES
        - value    An Epoch or list of Epochs that need to be converted.
        - unit     The units to use for an axis with Epoch data.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        return value.julianDate(unit) - cls.jdRef

    @classmethod
    def duration2float(cls, value):
        """: Convert a Duration value to a float suitable for plotting as a
              python datetime object.

        = INPUT VARIABLES
        - value    A Duration or list of Durations that need to be converted.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        return value.seconds() / cls.day_in_seconds

    @classmethod
    def to_numeric(cls, value, unit, axis):
        """: Convert value using unit to a float.  If value is a sequence, return
        the converted sequence.

        = INPUT VARIABLES
        - value    The value or list of values that need to be converted.
        - unit     The units to use for an axis with Epoch data.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        if not cbook.is_scalar_or_string(value):
            return [cls.to_numeric(x, unit, axis) for x in value]
        if cls.is_numlike(value):
            return value
        if unit is None:
            unit = cls.default_units(value, axis)
        if isinstance(value, U.Duration):
            return cls.duration2float(value)
        else:
            return cls.epoch2float(value, unit)

    @classmethod
    def from_numeric(cls, value, unit, axis):
        return date_ticker.num2date(value, unit)

    @classmethod
    def default_units(cls, value, axis):
        """: Return the default unit for value, or None.

        = INPUT VARIABLES
        - value    The value or list of values that need units.

        = RETURN VALUE
        - Returns the default units to use for value.
        """
        if cbook.is_scalar_or_string(value):
            return value.frame()
        else:
            return cls.default_units(value[0], axis)
