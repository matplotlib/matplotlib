"""UnitDblConverter module containing class UnitDblConverter."""

import numpy as np

import matplotlib.units as units
import matplotlib.projections.polar as polar

__all__ = ['UnitDblConverter']


# A special function for use with the matplotlib FuncFormatter class
# for formatting axes with radian units.
# This was copied from matplotlib example code.
def rad_fn(x, pos=None):
    """Radian function formatter."""
    n = int((x / np.pi) * 2.0 + 0.25)
    if n == 0:
        return str(x)
    elif n == 1:
        return r'$\pi/2$'
    elif n == 2:
        return r'$\pi$'
    elif n % 2 == 0:
        return fr'${n//2}\pi$'
    else:
        return fr'${n}\pi/2$'


class UnitDblConverter(units.ConversionInterface):
    """: A matplotlib converter class.  Provides matplotlib conversion
          functionality for the Monte UnitDbl class.
    """
    # default for plotting
    defaults = {
       "distance": 'km',
       "angle": 'deg',
       "time": 'sec',
       }

    @staticmethod
    def axisinfo(unit, axis):
        """: Returns information on how to handle an axis that has Epoch data.

        = INPUT VARIABLES
        - unit     The units to use for a axis with Epoch data.

        = RETURN VALUE
        - Returns a matplotlib AxisInfo data structure that contains
          minor/major formatters, major/minor locators, and default
          label information.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        # Check to see if the value used for units is a string unit value
        # or an actual instance of a UnitDbl so that we can use the unit
        # value for the default axis label value.
        if unit:
            label = unit if isinstance(unit, str) else unit.label()
        else:
            label = None

        if label == "deg" and isinstance(axis.axes, polar.PolarAxes):
            # If we want degrees for a polar plot, use the PolarPlotFormatter
            majfmt = polar.PolarAxes.ThetaFormatter()
        else:
            majfmt = U.UnitDblFormatter(useOffset=False)

        return units.AxisInfo(majfmt=majfmt, label=label)

    @staticmethod
    def convert(value, unit, axis):
        """: Convert value using unit to a float.  If value is a sequence, return
        the converted sequence.

        = INPUT VARIABLES
        - value    The value or list of values that need to be converted.
        - unit     The units to use for a axis with Epoch data.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        isNotUnitDbl = True

        if np.iterable(value) and not isinstance(value, str):
            if len(value) == 0:
                return []
            else:
                return [UnitDblConverter.convert(x, unit, axis) for x in value]

        # We need to check to see if the incoming value is actually a
        # UnitDbl and set a flag.  If we get an empty list, then just
        # return an empty list.
        if isinstance(value, U.UnitDbl):
            isNotUnitDbl = False

        # If the incoming value behaves like a number, but is not a UnitDbl,
        # then just return it because we don't know how to convert it
        # (or it is already converted)
        if isNotUnitDbl and units.ConversionInterface.is_numlike(value):
            return value

        # If no units were specified, then get the default units to use.
        if unit is None:
            unit = UnitDblConverter.default_units(value, axis)

        # Convert the incoming UnitDbl value/values to float/floats
        if isinstance(axis.axes, polar.PolarAxes) and value.type() == "angle":
            # Guarantee that units are radians for polar plots.
            return value.convert("rad")

        return value.convert(unit)

    @staticmethod
    def default_units(value, axis):
        """: Return the default unit for value, or None.

        = INPUT VARIABLES
        - value    The value or list of values that need units.

        = RETURN VALUE
        - Returns the default units to use for value.
        Return the default unit for value, or None.
        """

        # Determine the default units based on the user preferences set for
        # default units when printing a UnitDbl.
        if np.iterable(value) and not isinstance(value, str):
            return UnitDblConverter.default_units(value[0], axis)
        else:
            return UnitDblConverter.defaults[value.type()]
