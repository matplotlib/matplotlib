'''tzinfo timezone information for Africa/Luanda.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Luanda(DstTzInfo):
    '''Africa/Luanda timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Luanda'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,5,25,23,7,56),
        ]

    _transition_info = [
i(3120,0,'AOT'),
i(3600,0,'WAT'),
        ]

Luanda = Luanda()

