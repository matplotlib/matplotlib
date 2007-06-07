import sys

def check_globals():
    for key in globals().keys():
        if key in dir(sys.modules["__builtin__"]):
            if globals()[key] != getattr(sys.modules["__builtin__"],key):
                print "'%s' was overridden in globals()."%key

print "before pylab import"
check_globals()
print

from pylab import *

print "after pylab import"
check_globals()
print
