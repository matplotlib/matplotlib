'''tzinfo timezone information for America/Curacao.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Curacao(DstTzInfo):
    '''America/Curacao timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Curacao'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,2,12,4,35,44),
d(1965,1,1,4,30,0),
        ]

    _transition_info = [
i(-16560,0,'LMT'),
i(-16200,0,'ANT'),
i(-14400,0,'AST'),
        ]

Curacao = Curacao()

