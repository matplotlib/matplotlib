'''tzinfo timezone information for America/El_Salvador.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class El_Salvador(DstTzInfo):
    '''America/El_Salvador timezone definition. See datetime.tzinfo for details'''

    zone = 'America/El_Salvador'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1921,1,1,5,56,48),
d(1987,5,3,6,0,0),
d(1987,9,27,5,0,0),
d(1988,5,1,6,0,0),
d(1988,9,25,5,0,0),
        ]

    _transition_info = [
i(-21420,0,'LMT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
i(-18000,3600,'CDT'),
i(-21600,0,'CST'),
        ]

El_Salvador = El_Salvador()

