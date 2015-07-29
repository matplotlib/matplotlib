import matplotlib.pyplot as plt
import numpy as np

def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1, e1)

t1 = np.arange(0.0, 5.0, .1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

plt.figure(1)
plt.subplot(211)
l = plt.plot(t1, f(t1), 'k^')
plt.setp(l, markerfacecolor='k', markeredgecolor='r')
plt.title('A tale of 2 subplots', fontsize=14, fontname='Courier')
plt.ylabel('Signal 1', fontsize=12)
plt.subplot(212)
l = plt.plot(t1, f(t1), 'k>')

plt.ylabel('Signal 2', fontsize=12)
plt.xlabel('time (s)', fontsize=12)

plt.show()
