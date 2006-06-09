'''
Demonstration of quiver2 function.
Warning: API and plotting methods may change.
When quiver2 stabilizes it will replace quiver, so the
name "quiver2" is temporary.

Known problem: the plot autoscaling does not take into account
the arrows, so those on the boundaries are often out of the picture.
This is *not* an easy problem to solve in a perfectly general way.

'''
from pylab import *

X,Y = meshgrid( arange(0,2*pi,.2),arange(0,2*pi,.2) )
U = cos(X)
V = sin(Y)

figure()
quiver2( U, V)
title('Minimal arguments, no kwargs')

figure()
quiver2( X, Y, U, V, units='width')
title('scales with plot width, not view')

figure()
quiver2( X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
            pivot='mid', color='r', units='inches' )
plot( X[::3, ::3], Y[::3, ::3], 'k.')
title("pivot='mid'; every third arrow; units='inches'")

figure()
quiver2( X, Y, U, V, U+V, units='x', pivot='tip')
plot(X, Y, 'k.')
title("scales with x view; pivot='tip'")

figure()
quiver2( X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
             color='r', units='x',
            linewidths=(2,), edgecolors=('k'), headaxislength=5 )
title("triangular head; scale with x view; black edges")


show()

