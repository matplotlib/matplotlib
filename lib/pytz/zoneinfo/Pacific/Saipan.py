'''tzinfo timezone information for Pacific/Saipan.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Saipan(DstTzInfo):
    '''Pacific/Saipan timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Saipan'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1969,9,30,15,0,0),
d(2000,12,22,14,0,0),
        ]

    _transition_info = [
i(32400,0,'MPT'),
i(36000,0,'MPT'),
i(36000,0,'ChST'),
        ]

Saipan = Saipan()

