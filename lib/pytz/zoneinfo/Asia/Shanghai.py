'''tzinfo timezone information for Asia/Shanghai.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Shanghai(DstTzInfo):
    '''Asia/Shanghai timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Shanghai'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1927,12,31,15,54,8),
d(1940,6,2,16,0,0),
d(1940,9,30,15,0,0),
d(1941,3,15,16,0,0),
d(1941,9,30,15,0,0),
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
i(29160,0,'LMT'),
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
i(32400,3600,'CDT'),
i(28800,0,'CST'),
i(32400,3600,'CDT'),
i(28800,0,'CST'),
        ]

Shanghai = Shanghai()

