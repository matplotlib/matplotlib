'''tzinfo timezone information for America/Virgin.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Virgin(DstTzInfo):
    '''America/Virgin timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Virgin'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,7,1,4,19,44),
        ]

    _transition_info = [
i(-15600,0,'LMT'),
i(-14400,0,'AST'),
        ]

Virgin = Virgin()

