'''tzinfo timezone information for America/Coral_Harbour.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Coral_Harbour(DstTzInfo):
    '''America/Coral_Harbour timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Coral_Harbour'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,4,14,8,0,0),
d(1918,10,31,7,0,0),
d(1940,9,29,6,0,0),
d(1942,2,9,8,0,0),
d(1945,8,14,23,0,0),
d(1945,9,30,7,0,0),
        ]

    _transition_info = [
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-18000,3600,'CWT'),
i(-18000,3600,'CPT'),
i(-18000,0,'EST'),
        ]

Coral_Harbour = Coral_Harbour()

