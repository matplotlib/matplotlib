import sys, time, os
from matplotlib.numerix import rand
from matplotlib.transforms import identity_transform, unit_bbox, Func, IDENTITY
from matplotlib.transforms import one, Point, Value, Bbox, get_bbox_transform


def rand_point():
    xy = rand(2)
    return Point( Value(xy[0]),  Value(xy[1]) )

def rand_bbox():
    ll = rand_point()
    ur = rand_point()
    return Bbox(ll, ur)

def rand_transform():
    b1 = rand_bbox()
    b2 = rand_bbox()
    return get_bbox_transform(b1, b2)

