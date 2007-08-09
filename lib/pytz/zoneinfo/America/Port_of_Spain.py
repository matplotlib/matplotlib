'''tzinfo timezone information for America/Port_of_Spain.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Port_of_Spain(DstTzInfo):
    '''America/Port_of_Spain timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Port_of_Spain'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,3,2,4,6,4),
        ]

    _transition_info = [
i(-14760,0,'LMT'),
i(-14400,0,'AST'),
        ]

Port_of_Spain = Port_of_Spain()

