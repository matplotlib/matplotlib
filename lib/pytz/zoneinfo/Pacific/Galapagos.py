'''tzinfo timezone information for Pacific/Galapagos.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Galapagos(DstTzInfo):
    '''Pacific/Galapagos timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Galapagos'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1931,1,1,5,58,24),
d(1986,1,1,5,0,0),
        ]

    _transition_info = [
i(-21480,0,'LMT'),
i(-18000,0,'ECT'),
i(-21600,0,'GALT'),
        ]

Galapagos = Galapagos()

