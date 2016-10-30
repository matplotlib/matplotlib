"""
Show animation of a double pendulum.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

g = 9.8  # acceleration due to gravity, in m/s^2
length1 = 1.0  # length of pendulum 1 in m
length2 = 1.0  # length of pendulum 2 in m
mass1 = 1.0  # mass of pendulum 1 in kg
mass2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):
    """Derivatives"""
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = ((mass1 + mass2) * length1 -
            mass2 * length1 * np.cos(del_) * np.cos(del_))
    dydx[1] = (mass2 * length1 * state[1] * state[1] * np.sin(del_) *
               np.cos(del_) +
               mass2 * g * np.sin(state[2]) * np.cos(del_) +
               mass2 * length2 * state[3] * state[3] * np.sin(del_) -
               (mass1 + mass2) * g * np.sin(state[0])) / den1

    dydx[2] = state[3]

    den2 = (length2 / length1) * den1
    dydx[3] = (-mass2 * length2 * state[3] * state[3] * np.sin(del_) *
               np.cos(del_) +
               (mass1 + mass2) * g * np.sin(state[0]) * np.cos(del_) -
               (mass1 + mass2) * length1 * state[1] * state[1] * np.sin(del_) -
               (mass1 + mass2) * g * np.sin(state[2])) / den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = length1 * np.sin(y[:, 0])
y1 = -length1 * np.cos(y[:, 0])

x2 = length2 * np.sin(y[:, 2]) + x1
y2 = -length2 * np.cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    """Create a line and text to show time of the animation."""
    line.set_data([], [])
    time_text.set_text('')
    plt.xlabel('horizontal position (m)')
    plt.ylabel('vertical position (m)')
    return line, time_text


def animate(i):
    """Update the animation.

    Updates the positions of the double pendulum and the time shown,
    indexed by i.
    """
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()
