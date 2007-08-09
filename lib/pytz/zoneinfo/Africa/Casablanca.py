'''tzinfo timezone information for Africa/Casablanca.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Casablanca(DstTzInfo):
    '''Africa/Casablanca timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Casablanca'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1913,10,26,0,30,20),
d(1939,9,12,0,0,0),
d(1939,11,18,23,0,0),
d(1940,2,25,0,0,0),
d(1945,11,17,23,0,0),
d(1950,6,11,0,0,0),
d(1950,10,28,23,0,0),
d(1967,6,3,12,0,0),
d(1967,9,30,23,0,0),
d(1974,6,24,0,0,0),
d(1974,8,31,23,0,0),
d(1976,5,1,0,0,0),
d(1976,7,31,23,0,0),
d(1977,5,1,0,0,0),
d(1977,9,27,23,0,0),
d(1978,6,1,0,0,0),
d(1978,8,3,23,0,0),
d(1984,3,16,0,0,0),
d(1985,12,31,23,0,0),
        ]

    _transition_info = [
i(-1800,0,'LMT'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,0,'CET'),
i(0,0,'WET'),
        ]

Casablanca = Casablanca()

