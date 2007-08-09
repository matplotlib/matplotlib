'''tzinfo timezone information for America/Tortola.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tortola(DstTzInfo):
    '''America/Tortola timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Tortola'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,7,1,4,18,28),
        ]

    _transition_info = [
i(-15480,0,'LMT'),
i(-14400,0,'AST'),
        ]

Tortola = Tortola()

