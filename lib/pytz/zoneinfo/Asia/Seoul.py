'''tzinfo timezone information for Asia/Seoul.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Seoul(DstTzInfo):
    '''Asia/Seoul timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Seoul'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1904,11,30,15,30,0),
d(1927,12,31,15,0,0),
d(1931,12,31,15,30,0),
d(1954,3,20,15,0,0),
d(1960,5,14,16,0,0),
d(1960,9,12,15,0,0),
d(1961,8,9,16,0,0),
d(1968,9,30,15,30,0),
d(1987,5,9,15,0,0),
d(1987,10,10,14,0,0),
d(1988,5,7,15,0,0),
d(1988,10,8,14,0,0),
        ]

    _transition_info = [
i(30600,0,'KST'),
i(32400,0,'KST'),
i(30600,0,'KST'),
i(32400,0,'KST'),
i(28800,0,'KST'),
i(32400,3600,'KDT'),
i(28800,0,'KST'),
i(30600,0,'KST'),
i(32400,0,'KST'),
i(36000,3600,'KDT'),
i(32400,0,'KST'),
i(36000,3600,'KDT'),
i(32400,0,'KST'),
        ]

Seoul = Seoul()

