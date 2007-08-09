'''tzinfo timezone information for Africa/Maputo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Maputo(DstTzInfo):
    '''Africa/Maputo timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Maputo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,21,49,40),
        ]

    _transition_info = [
i(7800,0,'LMT'),
i(7200,0,'CAT'),
        ]

Maputo = Maputo()

