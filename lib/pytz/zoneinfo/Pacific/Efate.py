'''tzinfo timezone information for Pacific/Efate.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Efate(DstTzInfo):
    '''Pacific/Efate timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Efate'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,12,12,46,44),
d(1983,9,24,13,0,0),
d(1984,3,24,12,0,0),
d(1984,10,22,13,0,0),
d(1985,3,23,12,0,0),
d(1985,9,28,13,0,0),
d(1986,3,22,12,0,0),
d(1986,9,27,13,0,0),
d(1987,3,28,12,0,0),
d(1987,9,26,13,0,0),
d(1988,3,26,12,0,0),
d(1988,9,24,13,0,0),
d(1989,3,25,12,0,0),
d(1989,9,23,13,0,0),
d(1990,3,24,12,0,0),
d(1990,9,22,13,0,0),
d(1991,3,23,12,0,0),
d(1991,9,28,13,0,0),
d(1992,1,25,12,0,0),
d(1992,10,24,13,0,0),
d(1993,1,23,12,0,0),
        ]

    _transition_info = [
i(40380,0,'LMT'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
i(43200,3600,'VUST'),
i(39600,0,'VUT'),
        ]

Efate = Efate()

