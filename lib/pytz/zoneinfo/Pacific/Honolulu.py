'''tzinfo timezone information for Pacific/Honolulu.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Honolulu(DstTzInfo):
    '''Pacific/Honolulu timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Honolulu'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1933,4,30,12,30,0),
d(1933,5,21,11,30,0),
d(1942,2,9,12,30,0),
d(1945,8,14,23,0,0),
d(1945,9,30,11,30,0),
d(1947,6,8,12,30,0),
        ]

    _transition_info = [
i(-37800,0,'HST'),
i(-34200,3600,'HDT'),
i(-37800,0,'HST'),
i(-34200,3600,'HWT'),
i(-34200,3600,'HPT'),
i(-37800,0,'HST'),
i(-36000,0,'HST'),
        ]

Honolulu = Honolulu()

