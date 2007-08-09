'''tzinfo timezone information for America/Lima.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Lima(DstTzInfo):
    '''America/Lima timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Lima'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1908,7,28,5,8,36),
d(1938,1,1,5,0,0),
d(1938,4,1,4,0,0),
d(1938,9,25,5,0,0),
d(1939,3,26,4,0,0),
d(1939,9,24,5,0,0),
d(1940,3,24,4,0,0),
d(1986,1,1,5,0,0),
d(1986,4,1,4,0,0),
d(1987,1,1,5,0,0),
d(1987,4,1,4,0,0),
d(1990,1,1,5,0,0),
d(1990,4,1,4,0,0),
d(1994,1,1,5,0,0),
d(1994,4,1,4,0,0),
        ]

    _transition_info = [
i(-18540,0,'LMT'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
i(-14400,3600,'PEST'),
i(-18000,0,'PET'),
        ]

Lima = Lima()

