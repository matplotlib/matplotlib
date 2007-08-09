'''tzinfo timezone information for America/Santo_Domingo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Santo_Domingo(DstTzInfo):
    '''America/Santo_Domingo timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Santo_Domingo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1933,4,1,16,40,0),
d(1966,10,30,5,0,0),
d(1967,2,28,4,0,0),
d(1969,10,26,5,0,0),
d(1970,2,21,4,30,0),
d(1970,10,25,5,0,0),
d(1971,1,20,4,30,0),
d(1971,10,31,5,0,0),
d(1972,1,21,4,30,0),
d(1972,10,29,5,0,0),
d(1973,1,21,4,30,0),
d(1973,10,28,5,0,0),
d(1974,1,21,4,30,0),
d(1974,10,27,5,0,0),
d(2000,10,29,6,0,0),
d(2000,12,3,6,0,0),
        ]

    _transition_info = [
i(-16800,0,'SDMT'),
i(-18000,0,'EST'),
i(-14400,3600,'EDT'),
i(-18000,0,'EST'),
i(-16200,1800,'EHDT'),
i(-18000,0,'EST'),
i(-16200,1800,'EHDT'),
i(-18000,0,'EST'),
i(-16200,1800,'EHDT'),
i(-18000,0,'EST'),
i(-16200,1800,'EHDT'),
i(-18000,0,'EST'),
i(-16200,1800,'EHDT'),
i(-18000,0,'EST'),
i(-14400,0,'AST'),
i(-18000,0,'EST'),
i(-14400,0,'AST'),
        ]

Santo_Domingo = Santo_Domingo()

