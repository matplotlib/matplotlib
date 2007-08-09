'''tzinfo timezone information for Asia/Ashgabat.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Ashgabat(DstTzInfo):
    '''Asia/Ashgabat timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Ashgabat'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1924,5,1,20,6,28),
d(1930,6,20,20,0,0),
d(1981,3,31,19,0,0),
d(1981,9,30,18,0,0),
d(1982,3,31,19,0,0),
d(1982,9,30,18,0,0),
d(1983,3,31,19,0,0),
d(1983,9,30,18,0,0),
d(1984,3,31,19,0,0),
d(1984,9,29,21,0,0),
d(1985,3,30,21,0,0),
d(1985,9,28,21,0,0),
d(1986,3,29,21,0,0),
d(1986,9,27,21,0,0),
d(1987,3,28,21,0,0),
d(1987,9,26,21,0,0),
d(1988,3,26,21,0,0),
d(1988,9,24,21,0,0),
d(1989,3,25,21,0,0),
d(1989,9,23,21,0,0),
d(1990,3,24,21,0,0),
d(1990,9,29,21,0,0),
d(1991,3,30,21,0,0),
d(1991,9,28,22,0,0),
d(1991,10,26,20,0,0),
d(1992,1,18,22,0,0),
        ]

    _transition_info = [
i(14040,0,'LMT'),
i(14400,0,'ASHT'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(21600,3600,'ASHST'),
i(18000,0,'ASHT'),
i(18000,0,'ASHST'),
i(14400,0,'ASHT'),
i(14400,0,'TMT'),
i(18000,0,'TMT'),
        ]

Ashgabat = Ashgabat()

