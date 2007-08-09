'''tzinfo timezone information for Pacific/Norfolk.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Norfolk(DstTzInfo):
    '''Pacific/Norfolk timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Norfolk'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1950,12,31,12,48,0),
        ]

    _transition_info = [
i(40320,0,'NMT'),
i(41400,0,'NFT'),
        ]

Norfolk = Norfolk()

