'''tzinfo timezone information for Asia/Jakarta.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Jakarta(DstTzInfo):
    '''Asia/Jakarta timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Jakarta'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1923,12,31,16,40,0),
d(1932,10,31,16,40,0),
d(1942,3,22,16,30,0),
d(1945,7,31,15,0,0),
d(1948,4,30,16,30,0),
d(1950,4,30,16,0,0),
d(1963,12,31,16,30,0),
        ]

    _transition_info = [
i(25620,0,'JMT'),
i(26400,0,'JAVT'),
i(27000,0,'WIT'),
i(32400,0,'JST'),
i(27000,0,'WIT'),
i(28800,0,'WIT'),
i(27000,0,'WIT'),
i(25200,0,'WIT'),
        ]

Jakarta = Jakarta()

