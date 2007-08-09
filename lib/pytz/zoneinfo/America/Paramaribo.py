'''tzinfo timezone information for America/Paramaribo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Paramaribo(DstTzInfo):
    '''America/Paramaribo timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Paramaribo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,1,1,3,40,40),
d(1935,1,1,3,40,52),
d(1945,10,1,3,40,36),
d(1975,11,20,3,30,0),
d(1984,10,1,3,30,0),
        ]

    _transition_info = [
i(-13260,0,'LMT'),
i(-13260,0,'PMT'),
i(-13260,0,'PMT'),
i(-12600,0,'NEGT'),
i(-12600,0,'SRT'),
i(-10800,0,'SRT'),
        ]

Paramaribo = Paramaribo()

