from matplotlib.backends.backend_gtk import show_xvfb
from matplotlib.matlab import *

t = arange(0.0, 1.0, 0.002)
s = sin(2*2*pi*t)
l = plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
t = title('About as simple as it gets, folks')


#grid(True)
#set(gca(), 'xticks', (0,.2,.7))
#savefig('test2', dpi=600)
#show_xvfb()
show()

