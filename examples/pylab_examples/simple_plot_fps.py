"""
Example: simple line plot.
Show how to make and save a simple line plot with labels, title and grid
"""
from __future__ import print_function  # not necessary in Python 3.x
import matplotlib.pyplot as plt
import numpy as np
import time


plt.ion()

t = np.arange(0.0, 1.0 + 0.001, 0.001)
s = np.cos(2*2*np.pi*t)
plt.plot(t, s, '-', lw=2)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)

frames = 100.0
t = time.time()
c = time.clock()
for i in range(int(frames)):
    part = i / frames
    plt.axis([0.0, 1.0 - part, -1.0 + part, 1.0 - part])
wallclock = time.time() - t
user = time.clock() - c
print("wallclock:", wallclock)
print("user:", user)
print("fps:", frames / wallclock)
