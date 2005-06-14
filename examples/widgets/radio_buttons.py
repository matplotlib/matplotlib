from matplotlib.widgets import RadioButtons
from pylab import *
t = arange(0.0, 2.0, 0.01)
s0 = sin(2*pi*t)
s1 = sin(4*pi*t)
s2 = sin(8*pi*t)

ax = subplot(111)
l, = ax.plot(t, s0, lw=2, color='red')
subplots_adjust(left=0.3)

rax = axes([0.05, 0.4, 0.175, 0.175])
radio = RadioButtons(rax, ('button 1', 'button 2', 'button 3'))


def func(label):
    if label=='button 1':
        ydata = s0
        color = 'red'
    elif label=='button 2':
        ydata = s1
        color='blue'        
    elif label=='button 3':
        ydata = s2
        color='green'

    l.set_ydata(ydata)
    l.set_color(color)
    draw()
radio.on_clicked(func)
    
show()
