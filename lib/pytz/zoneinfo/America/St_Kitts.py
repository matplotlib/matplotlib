'''tzinfo timezone information for America/St_Kitts.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class St_Kitts(DstTzInfo):
    '''America/St_Kitts timezone definition. See datetime.tzinfo for details'''

    zone = 'America/St_Kitts'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,3,2,4,10,52),
        ]

    _transition_info = [
i(-15060,0,'LMT'),
i(-14400,0,'AST'),
        ]

St_Kitts = St_Kitts()

