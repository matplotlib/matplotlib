'''tzinfo timezone information for America/St_Vincent.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class St_Vincent(DstTzInfo):
    '''America/St_Vincent timezone definition. See datetime.tzinfo for details'''

    zone = 'America/St_Vincent'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,4,4,56),
        ]

    _transition_info = [
i(-14700,0,'KMT'),
i(-14400,0,'AST'),
        ]

St_Vincent = St_Vincent()

