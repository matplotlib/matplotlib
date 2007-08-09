'''tzinfo timezone information for Antarctica/Mawson.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mawson(DstTzInfo):
    '''Antarctica/Mawson timezone definition. See datetime.tzinfo for details'''

    zone = 'Antarctica/Mawson'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1954,2,13,0,0,0),
        ]

    _transition_info = [
i(0,0,'zzz'),
i(21600,0,'MAWT'),
        ]

Mawson = Mawson()

