'''tzinfo timezone information for Antarctica/Vostok.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Vostok(DstTzInfo):
    '''Antarctica/Vostok timezone definition. See datetime.tzinfo for details'''

    zone = 'Antarctica/Vostok'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1957,12,16,0,0,0),
        ]

    _transition_info = [
i(0,0,'zzz'),
i(21600,0,'VOST'),
        ]

Vostok = Vostok()

