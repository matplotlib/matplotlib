from matplotlib.matlab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
l = plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
t = title('About as simple as it gets, folks', verticalalignment='bottom')
t.set_y(0.99)

#grid(True)
#set(gca(), 'xticks', (0,.2,.7))
savefig('test2.eps')
show()
