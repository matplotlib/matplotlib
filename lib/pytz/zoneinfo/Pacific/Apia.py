'''tzinfo timezone information for Pacific/Apia.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Apia(DstTzInfo):
    '''Pacific/Apia timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Apia'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,1,1,11,26,56),
d(1950,1,1,11,30,0),
        ]

    _transition_info = [
i(-41220,0,'LMT'),
i(-41400,0,'SAMT'),
i(-39600,0,'WST'),
        ]

Apia = Apia()

