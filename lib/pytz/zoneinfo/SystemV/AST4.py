'''tzinfo timezone information for SystemV/AST4.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class AST4(DstTzInfo):
    '''SystemV/AST4 timezone definition. See datetime.tzinfo for details'''

    _zone = 'SystemV/AST4'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1942,5,3,4,0,0),
d(1945,9,30,5,0,0),
        ]

    _transition_info = [
i(-14400,0,'AST'),
i(-10800,3600,'AWT'),
i(-14400,0,'AST'),
        ]

AST4 = AST4()

