'''tzinfo timezone information for America/Guatemala.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Guatemala(DstTzInfo):
    '''America/Guatemala timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Guatemala'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,10,5,6,2,4),
d(1973,11,25,6,0,0),
d(1974,2,24,5,0,0),
d(1983,5,21,6,0,0),
d(1983,9,22,5,0,0),
d(1991,3,23,6,0,0),
d(1991,9,7,5,0,0),
d(2006,4,30,6,0,0),
d(2006,10,1,5,0,0),
        ]

    _transition_info = [
i(-21720,0,'LMT'),
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

Guatemala = Guatemala()

