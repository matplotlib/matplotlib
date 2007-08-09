'''tzinfo timezone information for America/Managua.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Managua(DstTzInfo):
    '''America/Managua timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Managua'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1934,6,23,5,45,12),
d(1973,5,1,6,0,0),
d(1975,2,16,5,0,0),
d(1979,3,18,6,0,0),
d(1979,6,25,5,0,0),
d(1980,3,16,6,0,0),
d(1980,6,23,5,0,0),
d(1992,1,1,10,0,0),
d(1992,9,24,5,0,0),
d(1993,1,1,6,0,0),
d(1997,1,1,5,0,0),
d(2005,4,10,6,0,0),
d(2005,10,2,5,0,0),
d(2006,4,30,8,0,0),
d(2006,10,1,6,0,0),
        ]

    _transition_info = [
i(-20700,0,'MMT'),
i(-21600,0,'CST'),
i(-18000,0,'EST'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,0,'EST'),
i(-21600,0,'CST'),
i(-18000,0,'EST'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
        ]

Managua = Managua()

