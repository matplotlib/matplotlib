import sys, time, os
from matplotlib.numerix.mlab import rand
from matplotlib.transforms import identity_transform, unit_bbox, Func, IDENTITY
from matplotlib.transforms import one, Point, Value, Bbox, get_bbox_transform


def rand_val(N = 1):
    if N==1: return Value(rand())
    else: return [Value(val) for val in rand(N)]

def rand_point():
    return Point( rand_val(), rand_val() )

def rand_bbox():
    ll = rand_point()
    ur = rand_point()
    return Bbox(ll, ur)

def rand_transform():
    b1 = rand_bbox()
    b2 = rand_bbox()
    return get_bbox_transform(b1, b2)

