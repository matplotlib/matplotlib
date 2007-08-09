'''tzinfo timezone information for Africa/Timbuktu.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Timbuktu(DstTzInfo):
    '''Africa/Timbuktu timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Timbuktu'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,0,32,0),
d(1934,2,26,0,0,0),
d(1960,6,20,1,0,0),
        ]

    _transition_info = [
i(-1920,0,'LMT'),
i(0,0,'GMT'),
i(-3600,0,'WAT'),
i(0,0,'GMT'),
        ]

Timbuktu = Timbuktu()

