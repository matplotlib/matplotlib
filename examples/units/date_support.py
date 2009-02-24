import matplotlib
matplotlib.rcParams['units'] = True
from matplotlib.cbook import iterable, is_numlike
import matplotlib.units as units
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import datetime

class DateConverter(units.ConversionInterface):

    @staticmethod
    def axisinfo(unit, axis):
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

    @staticmethod
    def convert(value, unit, axis):
        if units.ConversionInterface.is_numlike(value): return value
        return dates.date2num(value)

    @staticmethod
    def default_units(x, axis):
        'return the default unit for x or None'
        return 'date'


units.registry[datetime.date] = DateConverter()
units.registry[datetime.datetime] = DateConverter()
