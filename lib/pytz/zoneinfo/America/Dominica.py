'''tzinfo timezone information for America/Dominica.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dominica(DstTzInfo):
    '''America/Dominica timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Dominica'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,7,1,4,6,36),
        ]

    _transition_info = [
i(-14760,0,'LMT'),
i(-14400,0,'AST'),
        ]

Dominica = Dominica()

