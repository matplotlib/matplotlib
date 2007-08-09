'''tzinfo timezone information for Pacific/Kosrae.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kosrae(DstTzInfo):
    '''Pacific/Kosrae timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Kosrae'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1969,9,30,13,0,0),
d(1998,12,31,12,0,0),
        ]

    _transition_info = [
i(39600,0,'KOST'),
i(43200,0,'KOST'),
i(39600,0,'KOST'),
        ]

Kosrae = Kosrae()

