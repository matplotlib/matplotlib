'''tzinfo timezone information for Pacific/Enderbury.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Enderbury(DstTzInfo):
    '''Pacific/Enderbury timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Enderbury'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1979,10,1,12,0,0),
d(1995,1,1,11,0,0),
        ]

    _transition_info = [
i(-43200,0,'PHOT'),
i(-39600,0,'PHOT'),
i(46800,0,'PHOT'),
        ]

Enderbury = Enderbury()

