'''tzinfo timezone information for America/St_Thomas.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class St_Thomas(DstTzInfo):
    '''America/St_Thomas timezone definition. See datetime.tzinfo for details'''

    zone = 'America/St_Thomas'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,7,1,4,19,44),
        ]

    _transition_info = [
i(-15600,0,'LMT'),
i(-14400,0,'AST'),
        ]

St_Thomas = St_Thomas()

