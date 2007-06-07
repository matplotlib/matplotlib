"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked wether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
from pylab import subplot, arange, plot, sin, cos, pi, show
def line_select_callback(event1, event2):
    'event1 and event2 are the press and release events'
    x1, y1 = event1.xdata, event1.ydata
    x2, y2 = event2.xdata, event2.ydata
    print "(%3.2f, %3.2f) --> (%3.2f, %3.2f)"%(x1,y1,x2,y2)
    print " The button you used were: ",event1.button, event2.button


current_ax=subplot(111)                    # make a new plotingrange
N=100000                                   # If N is large one can see improvement
x=10.0*arange(N)/(N-1)                     # by use blitting!

plot(x,sin(.2*pi*x),lw=3,c='b',alpha=.7)   # plot something
plot(x,cos(.2*pi*x),lw=3.5,c='r',alpha=.5)
plot(x,-sin(.2*pi*x),lw=3.5,c='g',alpha=.3)

print "\n      click  -->  release"

# drawtype is 'box' or 'line' or 'none'
LS = RectangleSelector(current_ax, line_select_callback,
                      drawtype='box',useblit=True)
show()
