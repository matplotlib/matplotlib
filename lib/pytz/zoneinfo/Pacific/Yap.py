'''tzinfo timezone information for Pacific/Yap.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Yap(DstTzInfo):
    '''Pacific/Yap timezone definition. See datetime.tzinfo for details'''

    _zone = 'Pacific/Yap'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1969,9,30,15,0,0),
        ]

    _transition_info = [
i(32400,0,'YAPT'),
i(36000,0,'YAPT'),
        ]

Yap = Yap()

