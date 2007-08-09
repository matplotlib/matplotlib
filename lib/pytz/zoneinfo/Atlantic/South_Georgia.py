'''tzinfo timezone information for Atlantic/South_Georgia.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class South_Georgia(StaticTzInfo):
    '''Atlantic/South_Georgia timezone definition. See datetime.tzinfo for details'''
    zone = 'Atlantic/South_Georgia'
    _utcoffset = timedelta(seconds=-7200)
    _tzname = 'GST'

South_Georgia = South_Georgia()

