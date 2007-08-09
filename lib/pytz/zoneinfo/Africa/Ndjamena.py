'''tzinfo timezone information for Africa/Ndjamena.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Ndjamena(DstTzInfo):
    '''Africa/Ndjamena timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Ndjamena'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,22,59,48),
d(1979,10,13,23,0,0),
d(1980,3,7,22,0,0),
        ]

    _transition_info = [
i(3600,0,'LMT'),
i(3600,0,'WAT'),
i(7200,3600,'WAST'),
i(3600,0,'WAT'),
        ]

Ndjamena = Ndjamena()

