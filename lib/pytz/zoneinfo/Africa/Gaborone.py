'''tzinfo timezone information for Africa/Gaborone.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Gaborone(DstTzInfo):
    '''Africa/Gaborone timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Gaborone'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1943,9,19,0,0,0),
d(1944,3,18,23,0,0),
        ]

    _transition_info = [
i(7200,0,'CAT'),
i(10800,3600,'CAST'),
i(7200,0,'CAT'),
        ]

Gaborone = Gaborone()

