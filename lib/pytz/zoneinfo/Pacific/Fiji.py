'''tzinfo timezone information for Pacific/Fiji.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Fiji(DstTzInfo):
    '''Pacific/Fiji timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Fiji'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1915,10,25,12,6,20),
d(1998,10,31,14,0,0),
d(1999,2,27,14,0,0),
d(1999,11,6,14,0,0),
d(2000,2,26,14,0,0),
        ]

    _transition_info = [
i(42840,0,'LMT'),
i(43200,0,'FJT'),
i(46800,3600,'FJST'),
i(43200,0,'FJT'),
i(46800,3600,'FJST'),
i(43200,0,'FJT'),
        ]

Fiji = Fiji()

