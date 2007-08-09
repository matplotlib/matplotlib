'''tzinfo timezone information for Pacific/Tongatapu.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tongatapu(DstTzInfo):
    '''Pacific/Tongatapu timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Tongatapu'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1940,12,31,11,40,0),
d(1999,10,6,13,0,0),
d(2000,3,18,13,0,0),
d(2000,11,4,13,0,0),
d(2001,1,27,12,0,0),
d(2001,11,3,13,0,0),
d(2002,1,26,12,0,0),
        ]

    _transition_info = [
i(44400,0,'TOT'),
i(46800,0,'TOT'),
i(50400,3600,'TOST'),
i(46800,0,'TOT'),
i(50400,3600,'TOST'),
i(46800,0,'TOT'),
i(50400,3600,'TOST'),
i(46800,0,'TOT'),
        ]

Tongatapu = Tongatapu()

