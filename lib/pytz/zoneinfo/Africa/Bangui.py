'''tzinfo timezone information for Africa/Bangui.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Bangui(DstTzInfo):
    '''Africa/Bangui timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Bangui'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,22,45,40),
        ]

    _transition_info = [
i(4440,0,'LMT'),
i(3600,0,'WAT'),
        ]

Bangui = Bangui()

