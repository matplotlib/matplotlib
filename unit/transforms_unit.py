from __future__ import division
import sys
from Numeric import arange, asarray
from matplotlib.transforms import Bound1D, Bound2D, Transform
from matplotlib.transforms import identity, log10, pow10
from matplotlib.transforms import bound2d_all
from matplotlib.transforms import transform_bound1d
from matplotlib.transforms import TransformSize, Points, Dots, \
     Centimeter, Inches, RefVal
from matplotlib.artist import DPI

def closeto(x,y):
    try: assert(abs(asarray(x)-asarray(y))<1e-10)
    except AssertionError:
        raise AssertionError('Failed: %s and %s not close' %(x,y))

# testing bounding boxes
bbox = Bound2D( 1,1,5,5)
bbox.x.update([1,2,3,4,5,6])
bbox.y.update([-1,2,3,4,5,6,12])

left, bottom, width, height = bbox.get_bounds()
top = bottom + height
right = left + width
assert(left==1)
assert(right==6)
assert(width==5)
assert(bottom==-1)
assert(top==12)
assert(height==13)

b = Bound1D(-1, 1)
closeto(b.scale(1.9).bounds(), (-1.9, 1.9) )
b = Bound1D(-1, 1)
closeto(b.scale(0.9).bounds(), (-0.9, 0.9) )
print 'passed bounding box tests ...'

# test positive bounding box
x = arange(-1.0001, 1.0, 0.01)
bpos = Bound1D(.1, 1, isPos=True)
bpos.update(x)
assert(bpos.min()>0)
bpos.is_positive(False)
bpos.update(x)
assert(bpos.min()==-1.0001)

print 'passed positive bounding box tests ...'

# testing transforms
i1 = Bound1D(0,1)
i2 = Bound1D(-6,6)
identityTrans = Transform()
linearTrans = Transform(i1,i2)
logTrans = Transform(Bound1D(0.1,1), i2, funcs=(log10, pow10))

scalar = 1
tup = 1,2,3
a = arange(0.0, 2.0, 0.1)

assert( identityTrans.positions( scalar ) == scalar )
assert( identityTrans.positions( tup )== tup )
assert( identityTrans.positions( a ) == a )
assert( identityTrans.scale( scalar ) == scalar )
assert( identityTrans.scale( tup ) == tup )
assert( identityTrans.scale( a ) == a )

assert( linearTrans.positions(1) == 6 )
assert( linearTrans.scale(1) == 12 ) 
print 'passed transform tests ... '

bound = Bound1D(0.25, 0.75)
trans = Transform( Bound1D(0,1), Bound1D(-6,6))
bout = transform_bound1d(bound, trans)
assert( bout.bounds() == (-3.0, 3.0) )
print 'passed bbox transform tests ... '

# testing bound2d_all
b1 = Bound2D(1, 1, 1, 1)
b2 = Bound2D(1.1, 1.1, 1, 1)
b3 = Bound2D(1.2, 1.2, 1, 1)
b4 = Bound2D(1.3, 1.3, 4, 1)

bbox = bound2d_all((b1,b2,b3,b4))

closeto( bbox.x.min(), 1)
closeto( bbox.x.max(), 5.3)
closeto( bbox.y.min(), 1)
closeto( bbox.y.max(), 2.3)
print 'passed bound2d_all tests ... '

# testing inverses
bpos = Bound1D(.1, 1, isPos=True)
trans = Transform( bpos, Bound1D(-6,6), funcs=(log10, pow10))
x = 0.2
closeto( trans.inverse_positions(trans.positions(x)), x ) 
closeto( trans.inverse_scale(trans.scale(x)), x ) 

trans.set_funcs( (identity, identity) ) 
closeto( trans.inverse_positions(trans.positions(x)), x ) 
closeto( trans.inverse_scale(trans.scale(x)), x ) 
print 'passed inverse transform tests ... '


dpi = DPI(100)
dots = Dots(dpi)
pts = Points(dpi)

trans = TransformSize(pts, dots, RefVal(10))
closeto(trans.positions(0), 10)
closeto(trans.positions(72), 110)

dpi.set(200)
closeto(trans.positions(0), 10)
closeto(trans.positions(72), 210)
closeto(trans.positions(36), 110)

cm = Centimeter(dpi)
inch = Inches(dpi)

trans = TransformSize(cm, inch, RefVal(0))
closeto(trans.positions(2), 2/2.54)
closeto(trans.inverse_positions(trans.positions(2)), 2)
#closeto(trans.positions(36), 110)

dpi.set(100)
pts = Points(dpi)
dots = Dots(dpi)

trans = TransformSize(pts, dots, RefVal(62.5))
closeto( trans.positions(12), 12/72*100 + 62.5)

trans = TransformSize(pts, dots, RefVal(10))
#print trans.positions(-12)

print 'passed size transform tests ... '


