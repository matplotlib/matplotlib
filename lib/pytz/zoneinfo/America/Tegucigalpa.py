'''tzinfo timezone information for America/Tegucigalpa.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tegucigalpa(DstTzInfo):
    '''America/Tegucigalpa timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Tegucigalpa'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1921,4,1,5,48,52),
d(1987,5,3,6,0,0),
d(1987,9,27,5,0,0),
d(1988,5,1,6,0,0),
d(1988,9,25,5,0,0),
d(2006,5,7,6,0,0),
d(2006,8,7,5,0,0),
d(2007,5,6,6,0,0),
d(2007,8,6,5,0,0),
d(2008,5,4,6,0,0),
d(2008,8,4,5,0,0),
d(2009,5,3,6,0,0),
d(2009,8,3,5,0,0),
        ]

    _transition_info = [
i(-20940,0,'LMT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
        ]

Tegucigalpa = Tegucigalpa()

