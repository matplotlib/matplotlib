'''tzinfo timezone information for Indian/Chagos.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Chagos(DstTzInfo):
    '''Indian/Chagos timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Chagos'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1906,12,31,19,10,20),
d(1995,12,31,19,0,0),
        ]

    _transition_info = [
i(17400,0,'LMT'),
i(18000,0,'IOT'),
i(21600,0,'IOT'),
        ]

Chagos = Chagos()

