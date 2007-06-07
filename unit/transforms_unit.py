#from __future__ import division

from matplotlib.numerix import array, asarray, alltrue, arange
from matplotlib.numerix.mlab import rand
from matplotlib.transforms import Point, Bbox, Value, Affine
from matplotlib.transforms import multiply_affines
from matplotlib.transforms import Func, IDENTITY, LOG10, POLAR, FuncXY
from matplotlib.transforms import SeparableTransformation
from matplotlib.transforms import identity_transform, unit_bbox
from matplotlib.transforms import get_bbox_transform
from matplotlib.transforms import transform_bbox, inverse_transform_bbox
from matplotlib.transforms import bbox_all
from matplotlib.transforms import copy_bbox_transform


def closeto(x,y):
    return abs(asarray(x)-asarray(y))<1e-10

def closeto_seq(xs,ys):
    return alltrue([closeto(x,y) for x,y in zip(xs, ys)])

def closeto_bbox(b1, b2):
    xmin1, xmax1 = b1.intervalx().get_bounds()
    ymin1, ymax1 = b1.intervaly().get_bounds()
    xmin2, xmax2 = b2.intervalx().get_bounds()
    ymin2, ymax2 = b2.intervaly().get_bounds()

    pairs = ( (xmin1, xmin2), (xmax1, xmax2), (ymin1, ymin2), (ymax1, ymax2))
    return alltrue([closeto(x,y) for x,y in pairs])

ll = Point( Value(10),  Value(10) )
ur = Point( Value(200), Value(40) )

bbox = Bbox(ll, ur)

assert(bbox.xmin()==10)
assert(bbox.width()==190)
assert(bbox.height()==30)

ll.x().set(12.0)
assert(bbox.xmin()==12)
assert(bbox.width()==188)
assert(bbox.height()==30)


a  = Value(10)
b  = Value(0)
c  = Value(0)
d  = Value(20)
tx = Value(-10)
ty = Value(-20)

affine = Affine(a,b,c,d,tx,ty)
# test transformation of xy tuple
x, y = affine.xy_tup( (10,20) )
assert(x==90)
assert(y==380)

# test transformation of sequence of xy tuples
xy = affine.seq_xy_tups( ( (10,20), (20,30), ) )
assert(xy[0] == (90, 380))
assert(xy[1] == (190, 580))

# test transformation of x and y sequences
xy = affine.seq_x_y(  (10,20), (20,30))
assert(xy[0] == (90, 190))
assert(xy[1] == (380, 580))

# test with numeric arrays
xy = affine.seq_x_y(  array((10,20)), array((20,30)))
assert(xy[0] == (90, 190))
assert(xy[1] == (380, 580))

# now change the x scale factor and make sure the affine updated
# properly
a.set(20)
xy = affine.seq_xy_tups( ( (10,20), (20,30), ) )
assert(xy[0] == (190, 380))
assert(xy[1] == (390, 580))

# Test the aritmetic operations on lazy values
v1 = Value(10)
v2 = Value(20)
o1 =  v1 + v2
assert( o1.get() == 30)

o2 =  v1 * v2
assert( o2.get() == 200)

v3 = Value(2)
o3 = (v1+v2)*v3
assert( o3.get() == 60)

# test a composition of affines
zero = Value(0)
one = Value(1)
two = Value(2)
num = Value(2)
a1 = Affine(num, zero, zero, num, zero, zero)
a2 = Affine(one, zero, zero, num, num, one )

pnt = 3,4
a = multiply_affines(a1, a2)
assert( a2.xy_tup(pnt) == (5,9) )
assert( a.xy_tup(pnt) == (10,18) )

a = multiply_affines(a2, a1)
assert( a1.xy_tup(pnt) == (6,8) )
assert( a.xy_tup(pnt) == (8,17) )


# change num to 4 and make sure the affine product is still right
num.set(4)
assert( a1.xy_tup(pnt) == (12,16) )
assert( a.xy_tup(pnt) == (16,65) )

# test affines with arithemtic sums of lazy values
val = num*(one + two)
a1 = Affine(one, zero, zero, val, num, val)
assert(a1.xy_tup(pnt) == (7, 60))

x = rand(20)
y = rand(20)
transform = identity_transform()
xout, yout = transform.seq_x_y(x,y)
assert((x,y) == transform.seq_x_y(x,y))


# test bbox transforms; transform the unit coordinate system to
# "display coords"
bboxin = unit_bbox()
ll = Point( Value(10),  Value(10) )
ur = Point( Value(200), Value(40) )
bboxout = Bbox(ll, ur)

transform = get_bbox_transform(bboxin, bboxout)

assert( transform.xy_tup( (0,0) )==(10, 10))
assert( transform.xy_tup( (1,1) )==(200, 40))
assert( transform.xy_tup( (0.5, 0.5) )==(105, 25))

# simulate a resize
ur.x().set(400)
ur.y().set(400)
assert( transform.xy_tup( (0,0) )==(10, 10))
assert( transform.xy_tup( (1,1) )==(400, 400))
assert( transform.xy_tup( (0.5, 0.5) )==(205, 205))

pairs = ( ( (0,   0  ), (10,  10 )  ),
          ( (1,   1  ), (400, 400) ),
          ( (0.5, 0.5), (205, 205) ) )

for p1, p2 in pairs:
    assert( closeto_seq( transform.xy_tup(p1), p2 ) )
    assert( closeto_seq( transform.inverse_xy_tup(p2), p1) )

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


transform = rand_transform()
transform.set_funcx(Func(LOG10))

x = rand(100)
y = rand(100)
xys = zip(x,y)
for xy in xys:
    xyt = transform.xy_tup(xy)
    xyi = transform.inverse_xy_tup(xyt)
    assert( closeto_seq(xy,xyi) )


ll = Point( Value(-10),  Value(-10) )
ur = Point( Value(200), Value(40) )
bbox = Bbox(ll, ur)
assert(bbox.xmin()==-10)
assert(bbox.xmax()==200)
assert(bbox.ymin()==-10)
assert(bbox.ymax()==40)

bbox.update(xys, False)  # don't ignore current lim

bbox.update(xys, True)  #ignore current lim
assert(bbox.xmin()==min(x))
assert(bbox.xmax()==max(x))
assert(bbox.ymin()==min(y))
assert(bbox.ymax()==max(y))


ll = Point( Value(-10),  Value(-10) )
ur = Point( Value(200), Value(40) )
bbox = Bbox(ll, ur)

ix = bbox.intervalx()
iy = bbox.intervaly()

assert(bbox.xmin()==-10)
assert(bbox.xmax()==200)
assert(bbox.ymin()==-10)
assert(bbox.ymax()==40)

ix.set_bounds(-30, 400)
assert(bbox.xmin()==-30)
assert(bbox.xmax()==400)
assert(bbox.ymin()==-10)
assert(bbox.ymax()==40)


num = Value(200.0)
den = Value(100.0)
div = num/den
assert(div.get()==2.0)


# test the inverse bbox functions
trans = rand_transform()
bbox1 = rand_bbox()
ibbox = inverse_transform_bbox(trans, bbox1)
bbox2 = transform_bbox(trans, ibbox)
assert(closeto_bbox(bbox1, bbox2))


ll = Point( Value(-10),  Value(-10) )
ur = Point( Value(200), Value(40) )
bbox = Bbox(ll, ur)
transform = get_bbox_transform(unit_bbox(), bbox)
assert( closeto_seq( inverse_transform_bbox(transform, bbox).get_bounds(),
                     (0,0,1,1)))
assert( closeto_seq( transform_bbox(transform, unit_bbox()).get_bounds(),
                     (-10,-10,210,50)))


# test the bbox all bounding functions
boxes = [rand_bbox() for i in range(20)]
xmin = min([box.xmin() for box in boxes])
xmax = max([box.xmax() for box in boxes])
ymin = min([box.ymin() for box in boxes])
ymax = max([box.ymax() for box in boxes])

box = bbox_all(boxes)
assert( closeto_seq( box.get_bounds(), (xmin, ymin, xmax-xmin, ymax-ymin)))




t1 = rand_transform()
oboundsx = t1.get_bbox1().intervalx().get_bounds()
oboundsy = t1.get_bbox1().intervaly().get_bounds()
t2 = copy_bbox_transform(t1)
t1.get_bbox1().intervalx().set_bounds(1,2)
t2.get_bbox2().intervaly().set_bounds(-1,12)
newboundsx = t2.get_bbox1().intervalx().get_bounds()
newboundsy = t2.get_bbox1().intervaly().get_bounds()
assert(oboundsx==newboundsx)
assert(oboundsy==newboundsy)


import math
polar = FuncXY(POLAR)
assert( closeto_seq( polar.map(math.pi,1), (-1,0)) )
assert( closeto_seq( polar.inverse(1,1), ( (math.pi/4), math.sqrt(2))) )



# This unit test requires "nan", which numarray.ieeespecial
# exports.  (But we can keep using the numerix module.)
try:
    from numarray.ieeespecial import nan
    have_nan = True
except ImportError:
    have_nan = False

if have_nan:
    y1=array([  2,nan,1,2,3,4])
    y2=array([nan,nan,1,2,3,4])

    x1=arange(len(y1))
    x2=arange(len(y2))

    bbox1 = Bbox(Point(Value(0),Value(0)),
                 Point(Value(1),Value(1)))

    bbox2 = Bbox(Point(Value(0),Value(0)),
                 Point(Value(1),Value(1)))

    bbox1.update_numerix(x1,y1,1)
    bbox2.update_numerix(x2,y2,1)

    assert( closeto_seq( bbox1.get_bounds(), bbox2.get_bounds() ) )
else:
    print 'nan could not be imported from numarray.ieeespecial, test skipped'

print 'all tests passed'
