'''tzinfo timezone information for Pacific/Rarotonga.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Rarotonga(DstTzInfo):
    '''Pacific/Rarotonga timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Rarotonga'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1978,11,12,10,30,0),
d(1979,3,4,9,30,0),
d(1979,10,28,10,0,0),
d(1980,3,2,9,30,0),
d(1980,10,26,10,0,0),
d(1981,3,1,9,30,0),
d(1981,10,25,10,0,0),
d(1982,3,7,9,30,0),
d(1982,10,31,10,0,0),
d(1983,3,6,9,30,0),
d(1983,10,30,10,0,0),
d(1984,3,4,9,30,0),
d(1984,10,28,10,0,0),
d(1985,3,3,9,30,0),
d(1985,10,27,10,0,0),
d(1986,3,2,9,30,0),
d(1986,10,26,10,0,0),
d(1987,3,1,9,30,0),
d(1987,10,25,10,0,0),
d(1988,3,6,9,30,0),
d(1988,10,30,10,0,0),
d(1989,3,5,9,30,0),
d(1989,10,29,10,0,0),
d(1990,3,4,9,30,0),
d(1990,10,28,10,0,0),
d(1991,3,3,9,30,0),
        ]

    _transition_info = [
i(-37800,0,'CKT'),
i(-34200,3600,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
i(-34200,1800,'CKHST'),
i(-36000,0,'CKT'),
        ]

Rarotonga = Rarotonga()

