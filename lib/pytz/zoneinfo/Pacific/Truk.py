'''tzinfo timezone information for Pacific/Truk.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Truk(StaticTzInfo):
    '''Pacific/Truk timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Truk'
    _utcoffset = timedelta(seconds=36000)
    _tzname = 'TRUT'

Truk = Truk()

