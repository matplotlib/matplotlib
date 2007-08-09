'''tzinfo timezone information for Pacific/Pitcairn.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Pitcairn(DstTzInfo):
    '''Pacific/Pitcairn timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Pitcairn'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1998,4,27,8,30,0),
        ]

    _transition_info = [
i(-30600,0,'PNT'),
i(-28800,0,'PST'),
        ]

Pitcairn = Pitcairn()

