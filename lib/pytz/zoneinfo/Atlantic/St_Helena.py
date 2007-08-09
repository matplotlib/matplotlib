'''tzinfo timezone information for Atlantic/St_Helena.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class St_Helena(DstTzInfo):
    '''Atlantic/St_Helena timezone definition. See datetime.tzinfo for details'''

    zone = 'Atlantic/St_Helena'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1951,1,1,0,22,48),
        ]

    _transition_info = [
i(-1380,0,'JMT'),
i(0,0,'GMT'),
        ]

St_Helena = St_Helena()

