'''tzinfo timezone information for Africa/Tunis.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tunis(DstTzInfo):
    '''Africa/Tunis timezone definition. See datetime.tzinfo for details'''

    _zone = 'Africa/Tunis'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,3,10,23,50,39),
d(1939,4,15,22,0,0),
d(1939,11,18,22,0,0),
d(1940,2,25,22,0,0),
d(1941,10,5,22,0,0),
d(1942,3,8,23,0,0),
d(1942,11,2,1,0,0),
d(1943,3,29,1,0,0),
d(1943,4,17,0,0,0),
d(1943,4,25,1,0,0),
d(1943,10,4,0,0,0),
d(1944,4,3,1,0,0),
d(1944,10,7,22,0,0),
d(1945,4,2,1,0,0),
d(1945,9,15,22,0,0),
d(1977,4,29,23,0,0),
d(1977,9,23,23,0,0),
d(1978,4,30,23,0,0),
d(1978,9,30,23,0,0),
d(1988,5,31,23,0,0),
d(1988,9,24,23,0,0),
d(1989,3,25,23,0,0),
d(1989,9,23,23,0,0),
d(1990,4,30,23,0,0),
d(1990,9,29,23,0,0),
        ]

    _transition_info = [
i(540,0,'PMT'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
        ]

Tunis = Tunis()

