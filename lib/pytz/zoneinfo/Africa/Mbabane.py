'''tzinfo timezone information for Africa/Mbabane.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mbabane(DstTzInfo):
    '''Africa/Mbabane timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Mbabane'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,21,55,36),
        ]

    _transition_info = [
i(7440,0,'LMT'),
i(7200,0,'SAST'),
        ]

Mbabane = Mbabane()

