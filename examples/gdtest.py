import matplotlib
matplotlib.use('GD')
from matplotlib.matlab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, .1)
t2 = arange(0.0, 5.0, 0.02)
t3 = arange(0.0, 2.0, 0.01)


if 1:
    subplot(211)
    l = plot(t1, f(t1), 'k-^')
    set(l, 'markerfacecolor', 'r')
    set(gca(), 'xlim', [0,5])
    #set(l, 'markeredgecolor', 'r')
    title('A tale of 2 subplots', fontsize=12)
    ylabel('Signal 1', fontsize=10)

    subplot(212)
    l = plot(t1, f(t1), 'k->')
    set(gca(), 'xlim', [0,5])
    ylabel('Signal 2', fontsize=10)
    xlabel('time (s)', fontsize=10,  fontname='Courier')

ax = gca()


#savefig('gdtest', dpi=150)
show()
