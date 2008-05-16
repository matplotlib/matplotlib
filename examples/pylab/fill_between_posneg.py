#!/usr/bin/env python
"""
From: James Boyle <boyle5@llnl.gov>
Subject: possible candidate for examples directory using fill
To: John Hunter <jdhunter@nitace.bsd.uchicago.edu>
Date: Tue, 8 Mar 2005 15:44:11 -0800

I often compare the output from two sensors and I am interested in the
sign of the differences.

I wrote the enclosed code for find the polygons of the positive and
negative regions of the difference of two curves. It is written for
simple-minded straightforwardness rather than speed or elegance.
It is easy to fill in the two sets of polygons with contrasting colors.
For efficiency one could use collections but my curves are such that
fill is quick enough.

The code uses a simple linear technique to find the crossover point,
this too could be made more sophisticated if one desired.

I have found this code to be very handy for the comparisons I perform
-
maybe someone else would find it useful.

--Jim
"""

#!/usr/bin/env python

from pylab import *

def findZero(i,x,y1,y2):
     im1 = i-1
     m1 = (y1[i] - y1[im1])/(x[i] - x[im1])
     m2 = (y2[i] - y2[im1])/(x[i] - x[im1])
     b1 = y1[im1] - m1*x[im1]
     b2 = y2[im1] - m2*x[im1]
     xZero = (b1 - b2)/(m2 - m1)
     yZero = m1*xZero + b1
     return (xZero, yZero)

def posNegFill(x,y1,y2):
      diff = y2 - y1
      pos = []
      neg = []
      xx1 = [x[0]]
      xx2 = [x[0]]
      yy1 = [y1[0]]
      yy2 = [y2[0]]
      oldSign = (diff[0] < 0 )
      npts = x.shape[0]
      for i in range(1,npts):
          newSign = (diff[i] < 0)
          if newSign != oldSign:
              xz,yz = findZero(i,x,y1,y2)
              xx1.append(xz)
              yy1.append(yz)
              xx2.reverse()
              xx1.extend(xx2)
              yy2.reverse()
              yy1.extend(yy2)
              if oldSign:
                  neg.append( (xx1,yy1) )
              else:
                  pos.append( (xx1,yy1) )
              xx1 = [xz,x[i]]
              xx2 = [xz,x[i]]
              yy1 = [yz,y1[i]]
              yy2 = [yz,y2[i]]
              oldSign = newSign
          else:
              xx1.append( x[i])
              xx2.append( x[i])
              yy1.append(y1[i])
              yy2.append(y2[i])
              if i == npts-1:
                  xx2.reverse()
                  xx1.extend(xx2)
                  yy2.reverse()
                  yy1.extend(yy2)
                  if oldSign :
                      neg.append( (xx1,yy1) )
                  else:
                      pos.append( (xx1,yy1) )
      return pos,neg

x1 = arange(0, 2, 0.01)
y1 = sin(2*pi*x1)
y2 = sin(4*pi*x1)

# find positive and negative polygons of difference
pos,neg =  posNegFill(x1,y1,y2)
# positive y2 > y1 is blue
for x,y  in pos:
     p = fill(x,y,'b')

# negative Y2 < y1 is red
for x,y in neg:
     p = fill(x,y,'r')

show()
