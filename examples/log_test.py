from matplotlib.matlab import *

dt = 0.01
t = arange(dt, 20.0, dt)

semilogx(t, exp(-t/5.0))
grid(True)
#savefig('log_demo')
show()
