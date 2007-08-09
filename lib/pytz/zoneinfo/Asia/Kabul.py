'''tzinfo timezone information for Asia/Kabul.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kabul(DstTzInfo):
    '''Asia/Kabul timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Kabul'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1944,12,31,20,0,0),
        ]

    _transition_info = [
i(14400,0,'AFT'),
i(16200,0,'AFT'),
        ]

Kabul = Kabul()

