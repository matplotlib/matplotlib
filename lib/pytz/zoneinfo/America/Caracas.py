'''tzinfo timezone information for America/Caracas.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Caracas(DstTzInfo):
    '''America/Caracas timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Caracas'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,2,12,4,27,40),
d(1965,1,1,4,30,0),
        ]

    _transition_info = [
i(-16080,0,'CMT'),
i(-16200,0,'VET'),
i(-14400,0,'VET'),
        ]

Caracas = Caracas()

