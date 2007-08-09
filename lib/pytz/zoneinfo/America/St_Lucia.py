'''tzinfo timezone information for America/St_Lucia.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class St_Lucia(DstTzInfo):
    '''America/St_Lucia timezone definition. See datetime.tzinfo for details'''

    zone = 'America/St_Lucia'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,4,4,0),
        ]

    _transition_info = [
i(-14640,0,'CMT'),
i(-14400,0,'AST'),
        ]

St_Lucia = St_Lucia()

