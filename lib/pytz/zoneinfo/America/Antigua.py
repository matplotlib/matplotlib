'''tzinfo timezone information for America/Antigua.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Antigua(DstTzInfo):
    '''America/Antigua timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Antigua'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,3,2,4,7,12),
d(1951,1,1,5,0,0),
        ]

    _transition_info = [
i(-14820,0,'LMT'),
i(-18000,0,'EST'),
i(-14400,0,'AST'),
        ]

Antigua = Antigua()

