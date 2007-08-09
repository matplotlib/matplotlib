'''tzinfo timezone information for America/Belem.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Belem(DstTzInfo):
    '''America/Belem timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Belem'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1914,1,1,3,13,56),
d(1931,10,3,14,0,0),
d(1932,4,1,2,0,0),
d(1932,10,3,3,0,0),
d(1933,4,1,2,0,0),
d(1949,12,1,3,0,0),
d(1950,4,16,3,0,0),
d(1950,12,1,3,0,0),
d(1951,4,1,2,0,0),
d(1951,12,1,3,0,0),
d(1952,4,1,2,0,0),
d(1952,12,1,3,0,0),
d(1953,3,1,2,0,0),
d(1963,12,9,3,0,0),
d(1964,3,1,2,0,0),
d(1965,1,31,3,0,0),
d(1965,3,31,2,0,0),
d(1965,12,1,3,0,0),
d(1966,3,1,2,0,0),
d(1966,11,1,3,0,0),
d(1967,3,1,2,0,0),
d(1967,11,1,3,0,0),
d(1968,3,1,2,0,0),
d(1985,11,2,3,0,0),
d(1986,3,15,2,0,0),
d(1986,10,25,3,0,0),
d(1987,2,14,2,0,0),
d(1987,10,25,3,0,0),
d(1988,2,7,2,0,0),
        ]

    _transition_info = [
i(-11640,0,'LMT'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
i(-7200,3600,'BRST'),
i(-10800,0,'BRT'),
        ]

Belem = Belem()

