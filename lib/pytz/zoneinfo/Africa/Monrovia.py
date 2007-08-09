'''tzinfo timezone information for Africa/Monrovia.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Monrovia(DstTzInfo):
    '''Africa/Monrovia timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Monrovia'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,3,1,0,43,8),
d(1972,5,1,0,44,30),
        ]

    _transition_info = [
i(-2580,0,'MMT'),
i(-2640,0,'LRT'),
i(0,0,'GMT'),
        ]

Monrovia = Monrovia()

