# This file generates the matplotlib web page logo
from matplotlib.matlab import *


# convert data to mV
x = 1000*0.1*fromstring(
    file('data/membrane.dat', 'rb').read(), typecode=Float32)
# 0.0005 is the sample interval
t = 0.0005*arange(len(x))
figure(1, size=(700,100))
#subplot(111, axisbg='#afafaf')
subplot(111, axisbg=(205,179,139))
#subplot(111, axisbg='k')
plot(t, x)
text(1.1,-55,'matplotlib', color=(47,79,79), fontsize=50, fontname='Courier')
#text(1.1,-55,'matplotlib', color='b', fontsize=50, fontname='Courier')
axis([1, 1.72,-60, 10])
set(gca(), 'xticklabels', [])
set(gca(), 'yticklabels', [])
#savefig('logo2.png', dpi=300)
show()
