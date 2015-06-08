from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

import mplparsave

#### THIS CODE COMES FROM MATPLOTLIB EXAMPLES: ########

G =  9.8 # acceleration due to gravity, in m/s^2
L1 = 1.0 # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0 # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2]-state[0]
    den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
               + M2*G*sin(state[2])*cos(del_) + M2*L2*state[3]*state[3]*sin(del_)
               - (M1+M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)
               + (M1+M2)*G*sin(state[0])*cos(del_)
               - (M1+M2)*L1*state[1]*state[1]*sin(del_)
               - (M1+M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 40, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

rad = pi/180

# initial state
state = np.array([th1, w1, th2, w2])*pi/180.

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:,0])
y1 = -L1*cos(y[:,0])

x2 = L2*sin(y[:,2]) + x1
y2 = -L2*cos(y[:,2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

############## THIS CODE IS TO DEMONSTRATE HOW MPLPARSAVE WORKS ############

# The first half of the frames...
block1=np.arange(1, len(y)/2)
# The second half of the frames...
block2=np.arange(len(y)/2, len(y))
# Two blocks in total will tell the recorder to run two processes in
# parallel, one for each of the blocks, and stitch the two at the end.
# More blocks will trigger more processes. One shouldn't run more than
# the number of available cores.
blocks=[block1, block2]

# Set up the writer class (matplotlib proper). We use 'ffmpeg', but
# any of the supported writers can be used.
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, bitrate=1800)

# Record and stitch using ffmpeg as stitcher...
mplparsave.Parsave.record("ffmpeg-pendulum.mp4", fig, animate, init,
                          blocks, writer, mplparsave.Stitcher('ffmpeg'),
                          interval=35)

# Record and stitch using mencoder as stitcher...
mplparsave.Parsave.record("mencoder-pendulum.mp4", fig, animate, init,
                          blocks, writer, mplparsave.Stitcher('mencoder'),
                          interval=35)
