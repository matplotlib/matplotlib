'''tzinfo timezone information for Pacific/Palau.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Palau(StaticTzInfo):
    '''Pacific/Palau timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Palau'
    _utcoffset = timedelta(seconds=32400)
    _tzname = 'PWT'

Palau = Palau()

