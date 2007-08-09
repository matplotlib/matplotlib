'''tzinfo timezone information for Indian/Maldives.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Maldives(DstTzInfo):
    '''Indian/Maldives timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Maldives'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1959,12,31,19,6,0),
        ]

    _transition_info = [
i(17640,0,'MMT'),
i(18000,0,'MVT'),
        ]

Maldives = Maldives()

