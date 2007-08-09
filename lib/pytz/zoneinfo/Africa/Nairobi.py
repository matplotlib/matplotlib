'''tzinfo timezone information for Africa/Nairobi.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Nairobi(DstTzInfo):
    '''Africa/Nairobi timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Nairobi'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1928,6,30,21,32,44),
d(1929,12,31,21,0,0),
d(1939,12,31,21,30,0),
d(1959,12,31,21,15,15),
        ]

    _transition_info = [
i(8820,0,'LMT'),
i(10800,0,'EAT'),
i(9000,0,'BEAT'),
i(9900,0,'BEAUT'),
i(10800,0,'EAT'),
        ]

Nairobi = Nairobi()

