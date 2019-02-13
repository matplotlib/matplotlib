"""
Simulation of a free-cart pendulum

Parameters:
	L - length of the rod
	m - mass of the load on the top of the rod
	M - mass of the cart

System of equations of motion:
	L * th'' = -g * sin(th) + x'' * cos(th),
	(m + M) * x'' + m * th'' * L * cos(th) - m * L * Th'^2 * sin(th) = 0

System:
	th' = Y,
	Y' = (g * sin(th) + b * L * Y**2 * sin(th) * cos(th)) / (L * (1 + b * cos(th)**2)),
	x' = Z,
	Z' = b * (L * Y**2 * sin(th) - g * sin(th) * cos(th)) / (1 + b * cos(th)**2),
	where b = m / (M + m)

State:
	[th, Y, x, Z]

References:
	- (Original example)[https://matplotlib.org/gallery/animation/double_pendulum_sgskip.html]
"""

import numpy as np
import matplotlib.pyplot as pp
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from math import pi
from numpy import sin, cos

# physical constants
g = 9.8
L = 1.0
m = 0.5
M = 1.0
b = m / (m + M)

# simulation time
dt = 0.05
Tmax = 20
t = np.arange(0.0, Tmax, dt)

# initial conditions
Y = .0 		# pendulum angular velocity
th = pi/3	# pendulum angle
x = .0		# cart position
Z = .0		# cart velocity

state = np.array([th, Y, x, Z])

def derivatives(state, t):
	ds = np.zeros_like(state)

	ds[0] = state[1]
	ds[1] = (g * sin(state[0]) + b * L * state[1]**2 * sin(state[0]) * cos(state[0])) / (L * (1 + b * cos(state[0])**2))
	ds[2] = state[3]
	ds[3] = b * (L * state[1]**2 * sin(state[0]) - g * sin(state[0]) * cos(state[0])) / (1 + b * cos(state[0])**2)

	return ds

print("Integrating...")
# integrate your ODE using scipy.integrate.
solution = integrate.odeint(derivatives, state, t)
print("Done")

ths = solution[:, 0]
xs = solution[:, 2]

pxs = L * sin(ths) + xs
pys = L * cos(ths)

fig = pp.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

cart_width = 0.3
cart_height = 0.2

def init():
    line.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-cart_width/2, -cart_height/2))
    patch.set_width(cart_width)
    patch.set_height(cart_height)
    return line, time_text, patch


def animate(i):
    thisx = [xs[i], pxs[i]]
    thisy = [0, pys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    patch.set_x(xs[i] - cart_width/2)
    return line, time_text, patch

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(solution)),
                              interval=25, blit=True, init_func=init)

pp.show()

# Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Sergey Royz'), bitrate=1800)
# ani.save('free-cart.mp4', writer=writer)



