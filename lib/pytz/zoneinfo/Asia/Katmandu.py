'''tzinfo timezone information for Asia/Katmandu.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Katmandu(DstTzInfo):
    '''Asia/Katmandu timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Katmandu'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,18,18,44),
d(1985,12,31,18,30,0),
        ]

    _transition_info = [
i(20460,0,'LMT'),
i(19800,0,'IST'),
i(20700,0,'NPT'),
        ]

Katmandu = Katmandu()

