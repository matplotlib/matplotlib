'''tzinfo timezone information for Asia/Bangkok.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Bangkok(DstTzInfo):
    '''Asia/Bangkok timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Bangkok'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1920,3,31,17,17,56),
        ]

    _transition_info = [
i(24120,0,'BMT'),
i(25200,0,'ICT'),
        ]

Bangkok = Bangkok()

