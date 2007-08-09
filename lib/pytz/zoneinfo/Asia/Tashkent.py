'''tzinfo timezone information for Asia/Tashkent.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tashkent(DstTzInfo):
    '''Asia/Tashkent timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Tashkent'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1924,5,1,19,22,48),
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
d(1991,8,31,18,0,0),
d(1991,9,28,21,0,0),
d(1991,12,31,19,0,0),
        ]

    _transition_info = [
i(16620,0,'LMT'),
i(18000,0,'TAST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(25200,3600,'TASST'),
i(21600,0,'TAST'),
i(21600,0,'TASST'),
i(21600,0,'UZST'),
i(18000,0,'UZT'),
i(18000,0,'UZT'),
        ]

Tashkent = Tashkent()

