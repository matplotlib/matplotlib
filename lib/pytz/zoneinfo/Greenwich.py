'''tzinfo timezone information for Greenwich.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Greenwich(StaticTzInfo):
    '''Greenwich timezone definition. See datetime.tzinfo for details'''
    zone = 'Greenwich'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'GMT'

Greenwich = Greenwich()

