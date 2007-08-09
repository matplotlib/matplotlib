'''tzinfo timezone information for Asia/Kashgar.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kashgar(DstTzInfo):
    '''Asia/Kashgar timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Kashgar'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1927,12,31,18,56,4),
d(1939,12,31,18,30,0),
d(1980,4,30,19,0,0),
d(1986,5,3,16,0,0),
d(1986,9,13,15,0,0),
d(1987,4,11,16,0,0),
d(1987,9,12,15,0,0),
d(1988,4,9,16,0,0),
d(1988,9,10,15,0,0),
d(1989,4,15,16,0,0),
d(1989,9,16,15,0,0),
d(1990,4,14,16,0,0),
d(1990,9,15,15,0,0),
d(1991,4,13,16,0,0),
d(1991,9,14,15,0,0),
        ]

    _transition_info = [
i(18240,0,'LMT'),
i(19800,0,'KAST'),
i(18000,0,'KAST'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
        ]

Kashgar = Kashgar()

