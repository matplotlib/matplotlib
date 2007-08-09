'''tzinfo timezone information for Pacific/Wallis.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Wallis(StaticTzInfo):
    '''Pacific/Wallis timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Wallis'
    _utcoffset = timedelta(seconds=43200)
    _tzname = 'WFT'

Wallis = Wallis()

