'''tzinfo timezone information for America/Costa_Rica.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Costa_Rica(DstTzInfo):
    '''America/Costa_Rica timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Costa_Rica'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1921,1,15,5,36,20),
d(1979,2,25,6,0,0),
d(1979,6,3,5,0,0),
d(1980,2,24,6,0,0),
d(1980,6,1,5,0,0),
d(1991,1,19,6,0,0),
d(1991,7,1,5,0,0),
d(1992,1,18,6,0,0),
d(1992,3,15,5,0,0),
        ]

    _transition_info = [
i(-20160,0,'SJMT'),
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

Costa_Rica = Costa_Rica()

