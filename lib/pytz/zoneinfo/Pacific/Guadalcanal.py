'''tzinfo timezone information for Pacific/Guadalcanal.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Guadalcanal(DstTzInfo):
    '''Pacific/Guadalcanal timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Guadalcanal'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,9,30,13,20,12),
        ]

    _transition_info = [
i(38400,0,'LMT'),
i(39600,0,'SBT'),
        ]

Guadalcanal = Guadalcanal()

