from pylab import *
from matplotlib.widgets import CheckButtons

t = arange(0.0, 2.0, 0.01)
s0 = sin(2*pi*t)
s1 = sin(4*pi*t)
s2 = sin(6*pi*t)

ax = subplot(111)
l0, = ax.plot(t, s0, visible=False, lw=2)
l1, = ax.plot(t, s1, lw=2)
l2, = ax.plot(t, s2, lw=2)
subplots_adjust(left=0.2)

rax = axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))

def func(label):
    if label=='2 Hz': l0.set_visible(not l0.get_visible())
    elif label=='4 Hz': l1.set_visible(not l1.get_visible())
    elif label=='6 Hz': l2.set_visible(not l2.get_visible())
    draw()
check.on_clicked(func)

show()
