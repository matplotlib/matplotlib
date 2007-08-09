'''tzinfo timezone information for Asia/Jayapura.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Jayapura(DstTzInfo):
    '''Asia/Jayapura timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Jayapura'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1932,10,31,14,37,12),
d(1943,12,31,15,0,0),
d(1963,12,31,14,30,0),
        ]

    _transition_info = [
i(33780,0,'LMT'),
i(32400,0,'EIT'),
i(34200,0,'CST'),
i(32400,0,'EIT'),
        ]

Jayapura = Jayapura()

