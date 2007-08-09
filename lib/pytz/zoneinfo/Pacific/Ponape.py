'''tzinfo timezone information for Pacific/Ponape.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Ponape(StaticTzInfo):
    '''Pacific/Ponape timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Ponape'
    _utcoffset = timedelta(seconds=39600)
    _tzname = 'PONT'

Ponape = Ponape()

