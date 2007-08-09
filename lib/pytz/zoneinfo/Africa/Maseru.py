'''tzinfo timezone information for Africa/Maseru.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Maseru(DstTzInfo):
    '''Africa/Maseru timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Maseru'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,22,10,0),
d(1943,9,19,0,0,0),
d(1944,3,18,23,0,0),
        ]

    _transition_info = [
i(6600,0,'LMT'),
i(7200,0,'SAST'),
i(10800,3600,'SAST'),
i(7200,0,'SAST'),
        ]

Maseru = Maseru()

