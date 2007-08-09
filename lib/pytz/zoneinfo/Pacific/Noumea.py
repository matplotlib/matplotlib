'''tzinfo timezone information for Pacific/Noumea.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Noumea(DstTzInfo):
    '''Pacific/Noumea timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Noumea'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,12,12,54,12),
d(1977,12,3,13,0,0),
d(1978,2,26,12,0,0),
d(1978,12,2,13,0,0),
d(1979,2,26,12,0,0),
d(1996,11,30,15,0,0),
d(1997,3,1,15,0,0),
        ]

    _transition_info = [
i(39960,0,'LMT'),
i(39600,0,'NCT'),
i(43200,3600,'NCST'),
i(39600,0,'NCT'),
i(43200,3600,'NCST'),
i(39600,0,'NCT'),
i(43200,3600,'NCST'),
i(39600,0,'NCT'),
        ]

Noumea = Noumea()

