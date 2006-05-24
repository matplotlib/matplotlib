
from pylab import *

X,Y = meshgrid( arange(0,2*pi,.2),arange(0,2*pi,.2) )
U = cos(X)
V = sin(Y)

figure()
quiver( X, Y, U, V, 0.2, color='length')

figure()
quiver( U, V, 0.8, color='r' )

figure()
quiver( U, V)

figure()
quiver( U, V, color=U+V )

figure()
quiver( X, Y, U, V )
show()

