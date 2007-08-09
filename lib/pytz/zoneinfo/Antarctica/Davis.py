'''tzinfo timezone information for Antarctica/Davis.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Davis(DstTzInfo):
    '''Antarctica/Davis timezone definition. See datetime.tzinfo for details'''

    zone = 'Antarctica/Davis'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1957,1,13,0,0,0),
d(1964,10,31,17,0,0),
d(1969,2,1,0,0,0),
        ]

    _transition_info = [
i(0,0,'zzz'),
i(25200,0,'DAVT'),
i(0,0,'zzz'),
i(25200,0,'DAVT'),
        ]

Davis = Davis()

