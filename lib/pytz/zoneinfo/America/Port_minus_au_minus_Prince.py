'''tzinfo timezone information for America/Port_minus_au_minus_Prince.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Port_minus_au_minus_Prince(DstTzInfo):
    '''America/Port_minus_au_minus_Prince timezone definition. See datetime.tzinfo for details'''

    _zone = 'America/Port_minus_au_minus_Prince'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1917,1,24,16,49,0),
d(1983,5,8,5,0,0),
d(1983,10,30,4,0,0),
d(1984,4,29,5,0,0),
d(1984,10,28,4,0,0),
d(1985,4,28,5,0,0),
d(1985,10,27,4,0,0),
d(1986,4,27,5,0,0),
d(1986,10,26,4,0,0),
d(1987,4,26,5,0,0),
d(1987,10,25,4,0,0),
d(1988,4,3,6,0,0),
d(1988,10,30,6,0,0),
d(1989,4,2,6,0,0),
d(1989,10,29,6,0,0),
d(1990,4,1,6,0,0),
d(1990,10,28,6,0,0),
d(1991,4,7,6,0,0),
d(1991,10,27,6,0,0),
d(1992,4,5,6,0,0),
d(1992,10,25,6,0,0),
d(1993,4,4,6,0,0),
d(1993,10,31,6,0,0),
d(1994,4,3,6,0,0),
d(1994,10,30,6,0,0),
d(1995,4,2,6,0,0),
d(1995,10,29,6,0,0),
d(1996,4,7,6,0,0),
d(1996,10,27,6,0,0),
d(1997,4,6,6,0,0),
d(1997,10,26,6,0,0),
        ]

    _transition_info = [
i(-17340,0,'PPMT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
        ]

Port_minus_au_minus_Prince = Port_minus_au_minus_Prince()

