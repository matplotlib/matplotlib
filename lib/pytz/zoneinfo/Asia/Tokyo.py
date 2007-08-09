'''tzinfo timezone information for Asia/Tokyo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tokyo(DstTzInfo):
    '''Asia/Tokyo timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Tokyo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1937,12,31,15,0,0),
d(1948,5,1,17,0,0),
d(1948,9,10,16,0,0),
d(1949,4,2,17,0,0),
d(1949,9,9,16,0,0),
d(1950,5,6,17,0,0),
d(1950,9,8,16,0,0),
d(1951,5,5,17,0,0),
d(1951,9,7,16,0,0),
        ]

    _transition_info = [
i(32400,0,'CJT'),
i(32400,0,'JST'),
i(36000,3600,'JDT'),
i(32400,0,'JST'),
i(36000,3600,'JDT'),
i(32400,0,'JST'),
i(36000,3600,'JDT'),
i(32400,0,'JST'),
i(36000,3600,'JDT'),
i(32400,0,'JST'),
        ]

Tokyo = Tokyo()

