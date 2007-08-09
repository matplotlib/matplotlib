'''tzinfo timezone information for Pacific/Guam.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Guam(DstTzInfo):
    '''Pacific/Guam timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Guam'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(2000,12,22,14,0,0),
        ]

    _transition_info = [
i(36000,0,'GST'),
i(36000,0,'ChST'),
        ]

Guam = Guam()

