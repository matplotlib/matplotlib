'''tzinfo timezone information for America/Bogota.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Bogota(DstTzInfo):
    '''America/Bogota timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Bogota'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1914,11,23,4,56,20),
d(1992,5,3,5,0,0),
d(1993,4,4,4,0,0),
        ]

    _transition_info = [
i(-17760,0,'BMT'),
i(-18000,0,'COT'),
i(-14400,3600,'COST'),
i(-18000,0,'COT'),
        ]

Bogota = Bogota()

