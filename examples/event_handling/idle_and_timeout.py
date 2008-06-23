"""
Demonstrate/test the idle and timeout API
"""
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(10))

def on_idle(canvas):
    on_idle.count +=1
    print 'idle', on_idle.count
    if on_idle.count==10:
        canvas.mpl_source_remove(on_idle)
    return True
on_idle.count = 0

def on_timeout(canvas):
    on_timeout.count +=1
    print 'timeout', on_timeout.count
    if on_timeout.count==10:
        canvas.mpl_source_remove(on_timeout)
    return True
on_timeout.count = 0

fig.canvas.mpl_idle_add(on_idle)
fig.canvas.mpl_timeout_add(100, on_timeout)

plt.show()


