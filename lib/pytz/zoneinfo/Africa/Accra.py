'''tzinfo timezone information for Africa/Accra.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Accra(DstTzInfo):
    '''Africa/Accra timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Accra'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,1,1,0,0,52),
d(1936,9,1,0,0,0),
d(1936,12,30,23,40,0),
d(1937,9,1,0,0,0),
d(1937,12,30,23,40,0),
d(1938,9,1,0,0,0),
d(1938,12,30,23,40,0),
d(1939,9,1,0,0,0),
d(1939,12,30,23,40,0),
d(1940,9,1,0,0,0),
d(1940,12,30,23,40,0),
d(1941,9,1,0,0,0),
d(1941,12,30,23,40,0),
d(1942,9,1,0,0,0),
d(1942,12,30,23,40,0),
        ]

    _transition_info = [
i(-60,0,'LMT'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
i(1200,1200,'GHST'),
i(0,0,'GMT'),
        ]

Accra = Accra()

