'''tzinfo timezone information for Africa/Algiers.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Algiers(DstTzInfo):
    '''Africa/Algiers timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Algiers'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,3,10,23,50,39),
d(1916,6,14,23,0,0),
d(1916,10,1,23,0,0),
d(1917,3,24,23,0,0),
d(1917,10,7,23,0,0),
d(1918,3,9,23,0,0),
d(1918,10,6,23,0,0),
d(1919,3,1,23,0,0),
d(1919,10,5,23,0,0),
d(1920,2,14,23,0,0),
d(1920,10,23,23,0,0),
d(1921,3,14,23,0,0),
d(1921,6,21,23,0,0),
d(1939,9,11,23,0,0),
d(1939,11,19,0,0,0),
d(1940,2,25,2,0,0),
d(1944,4,3,1,0,0),
d(1944,10,8,0,0,0),
d(1945,4,2,1,0,0),
d(1945,9,15,23,0,0),
d(1946,10,6,23,0,0),
d(1956,1,29,0,0,0),
d(1963,4,13,23,0,0),
d(1971,4,25,23,0,0),
d(1971,9,26,23,0,0),
d(1977,5,6,0,0,0),
d(1977,10,20,23,0,0),
d(1978,3,24,0,0,0),
d(1978,9,22,1,0,0),
d(1979,10,25,23,0,0),
d(1980,4,25,0,0,0),
d(1980,10,31,1,0,0),
d(1981,5,1,0,0,0),
        ]

    _transition_info = [
i(540,0,'PMT'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(0,0,'WET'),
i(3600,0,'CET'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(3600,0,'CET'),
i(7200,3600,'CEST'),
i(3600,0,'CET'),
i(0,0,'WET'),
i(3600,3600,'WEST'),
i(0,0,'WET'),
i(3600,0,'CET'),
        ]

Algiers = Algiers()

