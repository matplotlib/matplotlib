'''tzinfo timezone information for Pacific/Marquesas.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Marquesas(DstTzInfo):
    '''Pacific/Marquesas timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Marquesas'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,10,1,9,18,0),
        ]

    _transition_info = [
i(-33480,0,'LMT'),
i(-34200,0,'MART'),
        ]

Marquesas = Marquesas()

