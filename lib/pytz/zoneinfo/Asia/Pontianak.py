'''tzinfo timezone information for Asia/Pontianak.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Pontianak(DstTzInfo):
    '''Asia/Pontianak timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Pontianak'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1908,4,30,16,42,40),
d(1932,10,31,16,42,40),
d(1942,1,28,16,30,0),
d(1945,7,31,15,0,0),
d(1948,4,30,16,30,0),
d(1950,4,30,16,0,0),
d(1963,12,31,16,30,0),
d(1987,12,31,16,0,0),
        ]

    _transition_info = [
i(26220,0,'LMT'),
i(26220,0,'PMT'),
i(27000,0,'WIT'),
i(32400,0,'JST'),
i(27000,0,'WIT'),
i(28800,0,'WIT'),
i(27000,0,'WIT'),
i(28800,0,'CIT'),
i(25200,0,'WIT'),
        ]

Pontianak = Pontianak()

