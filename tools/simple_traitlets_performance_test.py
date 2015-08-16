from __future__ import print_function  # not necessary in Python 3.x
import matplotlib.pyplot as plt
import numpy as np
import time

# test lifted directly from simple_plot_fps.py

def test():
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
    return dict([("wallclock", wallclock),
                 ("fps", frames / wallclock),
                 ("user", user)])

def ntest(n):

    totals = {"wallclock":0,
              "user":0,
              "fps":0}

    for i in range(n):
        t = test()
        for name in totals:
            totals[name] += t[name]
            
    for name in totals:
        totals[name] /= n
        
    return totals