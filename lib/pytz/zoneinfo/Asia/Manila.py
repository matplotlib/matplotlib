'''tzinfo timezone information for Asia/Manila.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Manila(DstTzInfo):
    '''Asia/Manila timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Manila'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1936,10,31,16,0,0),
d(1937,1,31,15,0,0),
d(1942,4,30,16,0,0),
d(1944,10,31,15,0,0),
d(1954,4,11,16,0,0),
d(1954,6,30,15,0,0),
d(1978,3,21,16,0,0),
d(1978,9,20,15,0,0),
        ]

    _transition_info = [
i(28800,0,'PHT'),
i(32400,3600,'PHST'),
i(28800,0,'PHT'),
i(32400,0,'JST'),
i(28800,0,'PHT'),
i(32400,3600,'PHST'),
i(28800,0,'PHT'),
i(32400,3600,'PHST'),
i(28800,0,'PHT'),
        ]

Manila = Manila()

