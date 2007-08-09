'''tzinfo timezone information for Africa/Kampala.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kampala(DstTzInfo):
    '''Africa/Kampala timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Kampala'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1928,6,30,21,50,20),
d(1929,12,31,21,0,0),
d(1947,12,31,21,30,0),
d(1956,12,31,21,15,15),
        ]

    _transition_info = [
i(7800,0,'LMT'),
i(10800,0,'EAT'),
i(9000,0,'BEAT'),
i(9900,0,'BEAUT'),
i(10800,0,'EAT'),
        ]

Kampala = Kampala()

