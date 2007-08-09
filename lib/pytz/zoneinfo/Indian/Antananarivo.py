'''tzinfo timezone information for Indian/Antananarivo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Antananarivo(DstTzInfo):
    '''Indian/Antananarivo timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Antananarivo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,6,30,20,49,56),
d(1954,2,27,20,0,0),
d(1954,5,29,20,0,0),
        ]

    _transition_info = [
i(11400,0,'LMT'),
i(10800,0,'EAT'),
i(14400,3600,'EAST'),
i(10800,0,'EAT'),
        ]

Antananarivo = Antananarivo()

