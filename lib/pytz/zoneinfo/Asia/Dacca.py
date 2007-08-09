'''tzinfo timezone information for Asia/Dacca.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dacca(DstTzInfo):
    '''Asia/Dacca timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Dacca'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1941,9,30,18,6,40),
d(1942,5,14,17,30,0),
d(1942,8,31,18,30,0),
d(1951,9,29,17,30,0),
d(1971,3,25,18,0,0),
        ]

    _transition_info = [
i(21180,0,'HMT'),
i(23400,0,'BURT'),
i(19800,0,'IST'),
i(23400,0,'BURT'),
i(21600,0,'DACT'),
i(21600,0,'BDT'),
        ]

Dacca = Dacca()

