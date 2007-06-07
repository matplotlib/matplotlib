import matplotlib
matplotlib.rcParams['units'] = True
from matplotlib.cbook import iterable, is_numlike
import matplotlib.units as units
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import datetime

class DateConverter(units.ConversionInterface):

    def axisinfo(unit):
        'return the unit AxisInfo'
        if unit=='date':
            majloc = dates.AutoDateLocator()
            majfmt = dates.AutoDateFormatter(majloc)
            return units.AxisInfo(
                majloc = majloc,
                majfmt = majfmt,
                label='date',
                )
        else: return None
    axisinfo = staticmethod(axisinfo)

    def convert(value, unit):
        if units.ConversionInterface.is_numlike(value): return value
        return dates.date2num(value)
    convert = staticmethod(convert)

    def default_units(x):
        'return the default unit for x or None'
        return 'date'
    default_units = staticmethod(default_units)


units.registry[datetime.date] = DateConverter()
units.registry[datetime.datetime] = DateConverter()
