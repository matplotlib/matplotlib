'''tzinfo timezone information for America/Aruba.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Aruba(DstTzInfo):
    '''America/Aruba timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Aruba'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,2,12,4,40,24),
d(1965,1,1,4,30,0),
        ]

    _transition_info = [
i(-16800,0,'LMT'),
i(-16200,0,'ANT'),
i(-14400,0,'AST'),
        ]

Aruba = Aruba()

