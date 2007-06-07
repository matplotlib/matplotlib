from pylab import *
from matplotlib.widgets import Button

freqs = arange(2,20,3)

ax = subplot(111)
subplots_adjust(bottom=0.2)
t = arange(0.0, 1.0, 0.001)
s = sin(2*pi*freqs[0]*t)
l, = plot(t,s, lw=2)


class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        i = self.ind%len(freqs)
        ydata = sin(2*pi*freqs[i]*t)
        l.set_ydata(ydata)
        draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind%len(freqs)
        ydata = sin(2*pi*freqs[i]*t)
        l.set_ydata(ydata)
        draw()

callback = Index()
axprev = axes([0.7, 0.05, 0.1, 0.075])
axnext = axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

show()

