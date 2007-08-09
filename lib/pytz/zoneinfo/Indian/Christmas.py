'''tzinfo timezone information for Indian/Christmas.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Christmas(StaticTzInfo):
    '''Indian/Christmas timezone definition. See datetime.tzinfo for details'''
    zone = 'Indian/Christmas'
    _utcoffset = timedelta(seconds=25200)
    _tzname = 'CXT'

Christmas = Christmas()

