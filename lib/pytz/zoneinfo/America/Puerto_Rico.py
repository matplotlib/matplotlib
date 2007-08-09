'''tzinfo timezone information for America/Puerto_Rico.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Puerto_Rico(DstTzInfo):
    '''America/Puerto_Rico timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Puerto_Rico'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1942,5,3,4,0,0),
d(1945,8,14,23,0,0),
d(1945,9,30,5,0,0),
        ]

    _transition_info = [
i(-14400,0,'AST'),
i(-10800,3600,'AWT'),
i(-10800,3600,'APT'),
i(-14400,0,'AST'),
        ]

Puerto_Rico = Puerto_Rico()

