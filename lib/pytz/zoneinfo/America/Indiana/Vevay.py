'''tzinfo timezone information for America/Indiana/Vevay.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Vevay(DstTzInfo):
    '''America/Indiana/Vevay timezone definition. See datetime.tzinfo for details'''

    _zone = 'America/Indiana/Vevay'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,3,31,8,0,0),
d(1918,10,27,7,0,0),
d(1919,3,30,8,0,0),
d(1919,10,26,7,0,0),
d(1942,2,9,8,0,0),
d(1945,8,14,23,0,0),
d(1945,9,30,7,0,0),
d(1954,4,25,8,0,0),
d(1969,4,27,7,0,0),
d(1969,10,26,6,0,0),
d(1970,4,26,7,0,0),
d(1970,10,25,6,0,0),
d(1971,4,25,7,0,0),
d(1971,10,31,6,0,0),
d(1972,4,30,7,0,0),
d(1972,10,29,6,0,0),
        ]

    _transition_info = [
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CWT'),
i(-18000,3600,'CPT'),
i(-21600,0,'CST'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
        ]

Vevay = Vevay()

