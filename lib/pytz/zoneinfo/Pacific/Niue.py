'''tzinfo timezone information for Pacific/Niue.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Niue(DstTzInfo):
    '''Pacific/Niue timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Niue'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1951,1,1,11,20,0),
d(1978,10,1,11,30,0),
        ]

    _transition_info = [
i(-40800,0,'NUT'),
i(-41400,0,'NUT'),
i(-39600,0,'NUT'),
        ]

Niue = Niue()

