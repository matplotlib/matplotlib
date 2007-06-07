import os, sys, time, gc

from matplotlib.numerix import array, asarray, alltrue
from matplotlib.numerix.mlab import rand
from matplotlib.transforms import Point, Bbox, Value, Affine
from matplotlib.transforms import multiply_affines
from matplotlib.transforms import Func, IDENTITY, LOG10, POLAR, FuncXY
from matplotlib.transforms import SeparableTransformation
from matplotlib.transforms import identity_transform, unit_bbox
from matplotlib.transforms import get_bbox_transform
from matplotlib.transforms import transform_bbox, inverse_transform_bbox
from matplotlib.transforms import bbox_all
from matplotlib.transforms import copy_bbox_transform, blend_xy_sep_transform


def report_memory(i):
    pid = os.getpid()
    if sys.platform=='sunos5':
        command = 'ps -p %d -o rss,osz' % pid
    else:
        'ps -p %d -o rss,sz' % pid
    a2 = os.popen(command).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[1])



# make some random bbox transforms and test inversion
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



class Line:
    def __init__(self):
        self._transform = identity_transform()

    def set_transform(self, t):
        self._transform = t

x, y = rand(2,10000)
indStart, indEnd = 30, 350
for i in range(indEnd):
    for j in range(20):
        l = Line()
        t1 = rand_transform()
        t2 = rand_transform()
        trans = blend_xy_sep_transform( t1, t2)
        l.set_transform(trans)
        xt, yt = trans.numerix_x_y(x, y)
        xytup = tuple(rand(2))
        txytup = trans.xy_tup(xytup)
        ixytup = trans.inverse_xy_tup(xytup)
        seqt = trans.seq_xy_tups(zip(x,y))
    gc.collect()
    val = report_memory(i)
    if i==indStart: start = val # wait a few cycles for memory usage to stabilize


end = val
print 'Average memory consumed per loop: %1.4fk bytes\n' % ((end-start)/float(indEnd-indStart))

