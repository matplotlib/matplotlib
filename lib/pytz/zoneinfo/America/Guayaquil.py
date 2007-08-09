'''tzinfo timezone information for America/Guayaquil.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Guayaquil(DstTzInfo):
    '''America/Guayaquil timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Guayaquil'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1931,1,1,5,14,0),
        ]

    _transition_info = [
i(-18840,0,'QMT'),
i(-18000,0,'ECT'),
        ]

Guayaquil = Guayaquil()

