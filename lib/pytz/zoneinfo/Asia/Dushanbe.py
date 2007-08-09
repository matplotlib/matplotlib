'''tzinfo timezone information for Asia/Dushanbe.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dushanbe(DstTzInfo):
    '''Asia/Dushanbe timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Dushanbe'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1924,5,1,19,24,48),
d(1930,6,20,19,0,0),
d(1981,3,31,18,0,0),
d(1981,9,30,17,0,0),
d(1982,3,31,18,0,0),
d(1982,9,30,17,0,0),
d(1983,3,31,18,0,0),
d(1983,9,30,17,0,0),
d(1984,3,31,18,0,0),
d(1984,9,29,20,0,0),
d(1985,3,30,20,0,0),
d(1985,9,28,20,0,0),
d(1986,3,29,20,0,0),
d(1986,9,27,20,0,0),
d(1987,3,28,20,0,0),
d(1987,9,26,20,0,0),
d(1988,3,26,20,0,0),
d(1988,9,24,20,0,0),
d(1989,3,25,20,0,0),
d(1989,9,23,20,0,0),
d(1990,3,24,20,0,0),
d(1990,9,29,20,0,0),
d(1991,3,30,20,0,0),
d(1991,9,8,21,0,0),
        ]

    _transition_info = [
i(16500,0,'LMT'),
i(18000,0,'DUST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(25200,3600,'DUSST'),
i(21600,0,'DUST'),
i(21600,0,'DUSST'),
i(18000,0,'TJT'),
        ]

Dushanbe = Dushanbe()

