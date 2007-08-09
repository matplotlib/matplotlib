'''tzinfo timezone information for Indian/Mauritius.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mauritius(DstTzInfo):
    '''Indian/Mauritius timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Mauritius'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1906,12,31,20,10,0),
        ]

    _transition_info = [
i(13800,0,'LMT'),
i(14400,0,'MUT'),
        ]

Mauritius = Mauritius()

