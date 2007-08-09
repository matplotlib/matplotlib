'''tzinfo timezone information for Pacific/Kiritimati.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kiritimati(DstTzInfo):
    '''Pacific/Kiritimati timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Kiritimati'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1979,10,1,10,40,0),
d(1995,1,1,10,0,0),
        ]

    _transition_info = [
i(-38400,0,'LINT'),
i(-36000,0,'LINT'),
i(50400,0,'LINT'),
        ]

Kiritimati = Kiritimati()

