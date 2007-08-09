'''tzinfo timezone information for America/La_Paz.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class La_Paz(DstTzInfo):
    '''America/La_Paz timezone definition. See datetime.tzinfo for details'''

    zone = 'America/La_Paz'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1931,10,15,4,32,36),
d(1932,3,21,3,32,36),
        ]

    _transition_info = [
i(-16380,0,'CMT'),
i(-12780,3600,'BOST'),
i(-14400,0,'BOT'),
        ]

La_Paz = La_Paz()

