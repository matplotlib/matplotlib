from matplotlib.matlab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .2)


l = plot(t1, f(t1), 'ro')
set(l, 'markersize', 30)
set(l, 'markerfacecolor', 'b')
savefig('arctest', dpi=150)
show()

