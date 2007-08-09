'''tzinfo timezone information for America/Martinique.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Martinique(DstTzInfo):
    '''America/Martinique timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Martinique'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,5,1,4,4,20),
d(1980,4,6,4,0,0),
d(1980,9,28,3,0,0),
        ]

    _transition_info = [
i(-14640,0,'FFMT'),
i(-14400,0,'AST'),
i(-10800,3600,'ADT'),
i(-14400,0,'AST'),
        ]

Martinique = Martinique()

