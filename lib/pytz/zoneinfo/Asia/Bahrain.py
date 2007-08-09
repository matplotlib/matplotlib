'''tzinfo timezone information for Asia/Bahrain.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Bahrain(DstTzInfo):
    '''Asia/Bahrain timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Bahrain'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,20,37,40),
d(1972,5,31,20,0,0),
        ]

    _transition_info = [
i(12120,0,'LMT'),
i(14400,0,'GST'),
i(10800,0,'AST'),
        ]

Bahrain = Bahrain()

