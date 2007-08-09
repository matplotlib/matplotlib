'''tzinfo timezone information for Asia/Qatar.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Qatar(DstTzInfo):
    '''Asia/Qatar timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Qatar'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,20,33,52),
d(1972,5,31,20,0,0),
        ]

    _transition_info = [
i(12360,0,'LMT'),
i(14400,0,'GST'),
i(10800,0,'AST'),
        ]

Qatar = Qatar()

