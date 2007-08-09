'''tzinfo timezone information for Asia/Harbin.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Harbin(DstTzInfo):
    '''Asia/Harbin timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Harbin'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1927,12,31,15,33,16),
d(1932,2,29,15,30,0),
d(1939,12,31,16,0,0),
d(1966,4,30,15,0,0),
d(1980,4,30,15,30,0),
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
i(30420,0,'LMT'),
i(30600,0,'CHAT'),
i(28800,0,'CST'),
i(32400,0,'CHAT'),
i(30600,0,'CHAT'),
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

Harbin = Harbin()

