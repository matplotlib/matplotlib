'''tzinfo timezone information for Africa/Dar_es_Salaam.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dar_es_Salaam(DstTzInfo):
    '''Africa/Dar_es_Salaam timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Dar_es_Salaam'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1930,12,31,21,22,52),
d(1947,12,31,21,0,0),
d(1960,12,31,21,15,15),
        ]

    _transition_info = [
i(9420,0,'LMT'),
i(10800,0,'EAT'),
i(9900,0,'BEAUT'),
i(10800,0,'EAT'),
        ]

Dar_es_Salaam = Dar_es_Salaam()

