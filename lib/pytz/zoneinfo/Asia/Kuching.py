'''tzinfo timezone information for Asia/Kuching.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kuching(DstTzInfo):
    '''Asia/Kuching timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Kuching'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1926,2,28,16,38,40),
d(1932,12,31,16,30,0),
d(1935,9,13,16,0,0),
d(1935,12,13,15,40,0),
d(1936,9,13,16,0,0),
d(1936,12,13,15,40,0),
d(1937,9,13,16,0,0),
d(1937,12,13,15,40,0),
d(1938,9,13,16,0,0),
d(1938,12,13,15,40,0),
d(1939,9,13,16,0,0),
d(1939,12,13,15,40,0),
d(1940,9,13,16,0,0),
d(1940,12,13,15,40,0),
d(1941,9,13,16,0,0),
d(1941,12,13,15,40,0),
d(1942,2,15,16,0,0),
d(1945,9,11,15,0,0),
d(1981,12,31,16,0,0),
        ]

    _transition_info = [
i(26460,0,'LMT'),
i(27000,0,'BORT'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(30000,1200,'BORTST'),
i(28800,0,'BORT'),
i(32400,0,'JST'),
i(28800,0,'BORT'),
i(28800,0,'MYT'),
        ]

Kuching = Kuching()

