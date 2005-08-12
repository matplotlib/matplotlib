'''This example is only interesting when ran in interactive mode'''

from pylab import *

# Plot circle or radius 3

an = linspace(0,2*pi,100)

subplot(221)
plot( 3*cos(an), 3*sin(an) )
title('not equal, looks like ellipse',fontsize=10)

subplot(222)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
title('equal, looks like circle',fontsize=10)

subplot(223)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
axis([-3,3,-3,3])
title('looks like circle, even after changing limits',fontsize=10)

subplot(224)
plot( 3*cos(an), 3*sin(an) )
axis('equal')
axis([-3,3,-3,3])
plot([0,4],[0,4])
title('still equal after adding line',fontsize=10)

show()





