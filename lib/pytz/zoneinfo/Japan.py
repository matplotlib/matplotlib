'''tzinfo timezone information for Japan.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Japan(DstTzInfo):
    '''Japan timezone definition. See datetime.tzinfo for details'''

    _zone = 'Japan'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1937,12,31,15,0,0),
        ]

    _transition_info = [
i(32400,0,'CJT'),
i(32400,0,'JST'),
        ]

Japan = Japan()

