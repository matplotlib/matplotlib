"""
Check to and form the epoch conversions for the various datetime
converters
"""
from matplotlib.dates import MxDatetimeConverter, PyDatetimeConverter,\
     EpochConverter
import mx.DateTime

dt1 = mx.DateTime.DateTime(2004, 03, 01)  # before dst
dt2 = mx.DateTime.DateTime(2004, 04, 15)  # after dst

dtc = MxDatetimeConverter()

assert( dtc.from_epoch( dtc.epoch(dt1) ) == dt1 )
assert( dtc.from_epoch( dtc.epoch(dt2) ) == dt2 )
print 'passed mx tests'

import datetime
dt1 = datetime.datetime(2004, 03, 01)  # before dst
dt2 = datetime.datetime(2004, 04, 15)  # after dst

dtc = PyDatetimeConverter()
assert( dtc.from_epoch( dtc.epoch(dt1) ) == dt1 )
assert( dtc.from_epoch( dtc.epoch(dt2) ) == dt2 )
print 'passed datetime tests'

# epoch
dt1 = 12345334
dt2 = 76543134

dtc = EpochConverter()
assert( dtc.from_epoch( dtc.epoch(dt1) ) == dt1 )
assert( dtc.from_epoch( dtc.epoch(dt2) ) == dt2 )
print 'passed epoch tests'
