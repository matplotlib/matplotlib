'''tzinfo timezone information for Antarctica/DumontDUrville.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class DumontDUrville(DstTzInfo):
    '''Antarctica/DumontDUrville timezone definition. See datetime.tzinfo for details'''

    zone = 'Antarctica/DumontDUrville'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1947,1,1,0,0,0),
d(1952,1,13,14,0,0),
d(1956,11,1,0,0,0),
        ]

    _transition_info = [
i(0,0,'zzz'),
i(36000,0,'PMT'),
i(0,0,'zzz'),
i(36000,0,'DDUT'),
        ]

DumontDUrville = DumontDUrville()

