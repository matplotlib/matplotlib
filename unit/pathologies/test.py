from Numeric import *

def is_string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1


y = arange(10)
print 'test1', is_string_like(y)

y.shape = 10,1
print 'test2', is_string_like(y)

y.shape = 1,10
print 'test3', is_string_like(y)
