'''tzinfo timezone information for Africa/Porto_minus_Novo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Porto_minus_Novo(DstTzInfo):
    '''Africa/Porto_minus_Novo timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Porto_minus_Novo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,23,49,32),
d(1934,2,26,0,0,0),
        ]

    _transition_info = [
i(600,0,'LMT'),
i(0,0,'GMT'),
i(3600,0,'WAT'),
        ]

Porto_minus_Novo = Porto_minus_Novo()

