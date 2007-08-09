'''tzinfo timezone information for Asia/Brunei.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Brunei(DstTzInfo):
    '''Asia/Brunei timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Brunei'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1926,2,28,16,20,20),
d(1932,12,31,16,30,0),
        ]

    _transition_info = [
i(27600,0,'LMT'),
i(27000,0,'BNT'),
i(28800,0,'BNT'),
        ]

Brunei = Brunei()

