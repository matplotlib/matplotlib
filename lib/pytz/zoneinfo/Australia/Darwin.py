'''tzinfo timezone information for Australia/Darwin.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Darwin(DstTzInfo):
    '''Australia/Darwin timezone definition. See datetime.tzinfo for details'''

    zone = 'Australia/Darwin'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1916,12,31,14,31,0),
d(1917,3,24,15,30,0),
d(1941,12,31,16,30,0),
d(1942,3,28,15,30,0),
d(1942,9,26,16,30,0),
d(1943,3,27,15,30,0),
d(1943,10,2,16,30,0),
d(1944,3,25,15,30,0),
        ]

    _transition_info = [
i(34200,0,'CST'),
i(37800,3600,'CST'),
i(34200,0,'CST'),
i(37800,3600,'CST'),
i(34200,0,'CST'),
i(37800,3600,'CST'),
i(34200,0,'CST'),
i(37800,3600,'CST'),
i(34200,0,'CST'),
        ]

Darwin = Darwin()

