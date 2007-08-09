'''tzinfo timezone information for Africa/Addis_Ababa.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Addis_Ababa(DstTzInfo):
    '''Africa/Addis_Ababa timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Addis_Ababa'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1936,5,4,21,24,40),
        ]

    _transition_info = [
i(9300,0,'ADMT'),
i(10800,0,'EAT'),
        ]

Addis_Ababa = Addis_Ababa()

