'''tzinfo timezone information for Pacific/Nauru.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Nauru(DstTzInfo):
    '''Pacific/Nauru timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Nauru'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1921,1,14,12,52,20),
d(1942,3,14,12,30,0),
d(1944,8,14,15,0,0),
d(1979,4,30,12,30,0),
        ]

    _transition_info = [
i(40080,0,'LMT'),
i(41400,0,'NRT'),
i(32400,0,'JST'),
i(41400,0,'NRT'),
i(43200,0,'NRT'),
        ]

Nauru = Nauru()

