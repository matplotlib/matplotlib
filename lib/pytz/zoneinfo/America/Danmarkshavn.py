'''tzinfo timezone information for America/Danmarkshavn.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Danmarkshavn(DstTzInfo):
    '''America/Danmarkshavn timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Danmarkshavn'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1916,7,28,1,14,40),
d(1980,4,6,5,0,0),
d(1980,9,28,1,0,0),
d(1981,3,29,1,0,0),
d(1981,9,27,1,0,0),
d(1982,3,28,1,0,0),
d(1982,9,26,1,0,0),
d(1983,3,27,1,0,0),
d(1983,9,25,1,0,0),
d(1984,3,25,1,0,0),
d(1984,9,30,1,0,0),
d(1985,3,31,1,0,0),
d(1985,9,29,1,0,0),
d(1986,3,30,1,0,0),
d(1986,9,28,1,0,0),
d(1987,3,29,1,0,0),
d(1987,9,27,1,0,0),
d(1988,3,27,1,0,0),
d(1988,9,25,1,0,0),
d(1989,3,26,1,0,0),
d(1989,9,24,1,0,0),
d(1990,3,25,1,0,0),
d(1990,9,30,1,0,0),
d(1991,3,31,1,0,0),
d(1991,9,29,1,0,0),
d(1992,3,29,1,0,0),
d(1992,9,27,1,0,0),
d(1993,3,28,1,0,0),
d(1993,9,26,1,0,0),
d(1994,3,27,1,0,0),
d(1994,9,25,1,0,0),
d(1995,3,26,1,0,0),
d(1995,9,24,1,0,0),
d(1996,1,1,3,0,0),
        ]

    _transition_info = [
i(-4500,0,'LMT'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(-7200,3600,'WGST'),
i(-10800,0,'WGT'),
i(0,0,'GMT'),
        ]

Danmarkshavn = Danmarkshavn()

