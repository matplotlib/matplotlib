#!/usr/bin/env python
# python

from matplotlib.matlab import *

dt = 0.01
t = arange(0,10,dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r, mode=2)*dt
cnse = cnse[:len(t)]
s = 0.1*sin(2*pi*t) + cnse

figure(1)
plot(t,s)

figure(2)
psd(s, 512, 1/dt)

#savefig('psd_demo.png')
show()


"""
% compare with matlab
dt = 0.01;
t = [0:dt:10];
nse = randn(size(t));
r = exp(-t/0.05);
cnse = conv(nse, r)*dt;
cnse = cnse(1:length(t));
s = 0.1*sin(2*pi*t) + cnse;
figure(1)
plot(t,s)

figure(2)
psd(s, 512, 1/dt)

"""
