'''tzinfo timezone information for America/Cayman.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Cayman(DstTzInfo):
    '''America/Cayman timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Cayman'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,2,1,5,7,12),
        ]

    _transition_info = [
i(-18420,0,'KMT'),
i(-18000,0,'EST'),
        ]

Cayman = Cayman()

