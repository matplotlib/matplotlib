'''tzinfo timezone information for America/Blanc_minus_Sablon.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Blanc_minus_Sablon(DstTzInfo):
    '''America/Blanc_minus_Sablon timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Blanc_minus_Sablon'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,4,14,6,0,0),
d(1918,10,31,5,0,0),
d(1942,2,9,6,0,0),
d(1945,8,14,23,0,0),
d(1945,9,30,5,0,0),
        ]

    _transition_info = [
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
i(-10800,3600,'AWT'),
i(-10800,3600,'APT'),
i(-14400,0,'AST'),
        ]

Blanc_minus_Sablon = Blanc_minus_Sablon()

