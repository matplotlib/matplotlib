'''tzinfo timezone information for Africa/Abidjan.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Abidjan(DstTzInfo):
    '''Africa/Abidjan timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Abidjan'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,0,16,8),
        ]

    _transition_info = [
i(-960,0,'LMT'),
i(0,0,'GMT'),
        ]

Abidjan = Abidjan()

