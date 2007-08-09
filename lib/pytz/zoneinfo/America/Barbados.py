'''tzinfo timezone information for America/Barbados.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Barbados(DstTzInfo):
    '''America/Barbados timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Barbados'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1924,1,1,3,58,28),
d(1932,1,1,3,58,28),
d(1977,6,12,6,0,0),
d(1977,10,2,5,0,0),
d(1978,4,16,6,0,0),
d(1978,10,1,5,0,0),
d(1979,4,15,6,0,0),
d(1979,9,30,5,0,0),
d(1980,4,20,6,0,0),
d(1980,9,25,5,0,0),
        ]

    _transition_info = [
i(-14280,0,'LMT'),
i(-14280,0,'BMT'),
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
        ]

Barbados = Barbados()

