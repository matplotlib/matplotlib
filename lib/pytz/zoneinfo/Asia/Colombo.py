'''tzinfo timezone information for Asia/Colombo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Colombo(DstTzInfo):
    '''Asia/Colombo timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Colombo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1905,12,31,18,40,28),
d(1942,1,4,18,30,0),
d(1942,8,31,18,0,0),
d(1945,10,15,19,30,0),
d(1996,5,24,18,30,0),
d(1996,10,25,18,0,0),
d(2006,4,14,18,30,0),
        ]

    _transition_info = [
i(19200,0,'MMT'),
i(19800,0,'IST'),
i(21600,1800,'IHST'),
i(23400,3600,'IST'),
i(19800,0,'IST'),
i(23400,0,'LKT'),
i(21600,0,'LKT'),
i(19800,0,'IST'),
        ]

Colombo = Colombo()

