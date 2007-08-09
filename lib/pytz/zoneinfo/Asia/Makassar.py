'''tzinfo timezone information for Asia/Makassar.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Makassar(DstTzInfo):
    '''Asia/Makassar timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Makassar'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,16,2,24),
d(1932,10,31,16,2,24),
d(1942,2,8,16,0,0),
d(1945,7,31,15,0,0),
        ]

    _transition_info = [
i(28680,0,'LMT'),
i(28680,0,'MMT'),
i(28800,0,'CIT'),
i(32400,0,'JST'),
i(28800,0,'CIT'),
        ]

Makassar = Makassar()

