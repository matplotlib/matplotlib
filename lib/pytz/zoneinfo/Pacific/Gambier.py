'''tzinfo timezone information for Pacific/Gambier.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Gambier(DstTzInfo):
    '''Pacific/Gambier timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Gambier'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,10,1,8,59,48),
        ]

    _transition_info = [
i(-32400,0,'LMT'),
i(-32400,0,'GAMT'),
        ]

Gambier = Gambier()

