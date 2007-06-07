from matplotlib.widgets import RadioButtons
from pylab import *
t = arange(0.0, 2.0, 0.01)
s0 = sin(2*pi*t)
s1 = sin(4*pi*t)
s2 = sin(8*pi*t)

ax = subplot(111)
l, = ax.plot(t, s0, lw=2, color='red')
subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = axes([0.05, 0.7, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('2 Hz', '4 Hz', '8 Hz'))
def hzfunc(label):
    hzdict = {'2 Hz':s0, '4 Hz':s1, '8 Hz':s2}
    ydata = hzdict[label]
    l.set_ydata(ydata)
    draw()
radio.on_clicked(hzfunc)

rax = axes([0.05, 0.4, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'))
def colorfunc(label):
    l.set_color(label)
    draw()
radio.on_clicked(colorfunc)

rax = axes([0.05, 0.1, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('-', '--', '-.', 'steps', ':'))
def stylefunc(label):
    l.set_linestyle(label)
    draw()
radio.on_clicked(stylefunc)

show()
