from matplotlib.matlab import *

figure(1)
t = arange(0.0, 1.0, 0.02)
s = sin(2*2*pi*t)
plot(t, s, 'x')

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)
savefig('simple_plot')
show()
