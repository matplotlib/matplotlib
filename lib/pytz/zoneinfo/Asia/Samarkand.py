'''tzinfo timezone information for Asia/Samarkand.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Samarkand(DstTzInfo):
    '''Asia/Samarkand timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Samarkand'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1924,5,1,19,32,48),
d(1930,6,20,20,0,0),
d(1981,3,31,19,0,0),
d(1981,9,30,18,0,0),
d(1982,3,31,18,0,0),
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
d(1991,8,31,18,0,0),
d(1991,9,28,21,0,0),
d(1991,12,31,19,0,0),
        ]

    _transition_info = [
i(16020,0,'LMT'),
i(14400,0,'SAMT'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(21600,0,'TAST'),
i(21600,0,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(18000,0,'SAMT'),
i(21600,3600,'SAMST'),
i(21600,3600,'UZST'),
i(18000,0,'UZT'),
i(18000,0,'UZT'),
        ]

Samarkand = Samarkand()

