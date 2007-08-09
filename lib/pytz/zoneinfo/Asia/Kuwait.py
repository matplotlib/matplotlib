'''tzinfo timezone information for Asia/Kuwait.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kuwait(DstTzInfo):
    '''Asia/Kuwait timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Kuwait'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1949,12,31,20,48,4),
        ]

    _transition_info = [
i(11520,0,'LMT'),
i(10800,0,'AST'),
        ]

Kuwait = Kuwait()

