
from pylab import *

X,Y = meshgrid( arange(0,2*pi,.2),arange(0,2*pi,.2) )
U = cos(X)
V = sin(Y)

quiver( X, Y, U, V, 0.2, color=True )
show()

quiver( U, V, 0.3 )
show()

quiver( U, V, color=True )
show()

quiver( U, V, color=U+V )
show()

quiver( X, Y, U, V )
show()

