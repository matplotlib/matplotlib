from pylab import *
from matplotlib.widgets import Slider, Button

ax = subplot(111)
subplots_adjust(bottom=0.25)
t = arange(0.0, 1.0, 0.001)
s = sin(2*pi*t)
l, = plot(t,s, lw=2)
axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = axes([0.125, 0.1, 0.775, 0.03], axisbg=axcolor)
axamp  = axes([0.125, 0.15, 0.775, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=1)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=1)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*sin(2*pi*freq*t))
    draw()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset')

def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)


show()

