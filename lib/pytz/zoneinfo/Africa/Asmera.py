'''tzinfo timezone information for Africa/Asmera.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Asmera(DstTzInfo):
    '''Africa/Asmera timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Asmera'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1936,5,4,21,24,40),
        ]

    _transition_info = [
i(9300,0,'ADMT'),
i(10800,0,'EAT'),
        ]

Asmera = Asmera()

