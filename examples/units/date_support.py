import matplotlib
matplotlib.rcParams['units'] = True

import matplotlib.units as units
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import basic_units
import datetime

class DateConverter(units.ConversionInterface):

    def tickers(x, unit=None):
        'return major and minor tick locators and formatters'
        majloc = dates.AutoDateLocator()
        minloc = ticker.NullLocator()
        majfmt = dates.AutoDateFormatter(majloc)
        minfmt = ticker.NullFormatter()
        return majloc, minloc, majfmt, minfmt
    tickers = staticmethod(tickers)

    def convert_to_value(value, unit):
        return dates.date2num(value)
    convert_to_value = staticmethod(convert_to_value)
        

units.manager.converters[datetime.date] = DateConverter()

