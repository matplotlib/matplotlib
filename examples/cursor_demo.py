#!/usr/bin/env python
"""

This example shows how to use matplotlib to provide a data cursor.  It
uses matplotlib to draw the cursor and may be a slow since this
requires redrawing the figure with every mouse move.

Faster cursoring is possible using native GUI drawing, as in
wxcursor_demo.py
"""
from pylab import *


class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx, = ax.plot( (0,0), (0,0), 'k-' )  # the horiz line
        self.ly, = ax.plot( (0,0), (0,0), 'k-' )  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes: return
        ax = event.inaxes
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_data( (minx, maxx), (y, y) )
        self.ly.set_data( (x, x), (miny, maxy) )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        draw()


class SnaptoCursor:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx, = ax.plot( (0,0), (0,0), 'k-' )  # the horiz line
        self.ly, = ax.plot( (0,0), (0,0), 'k-' )  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes: return
        ax = event.inaxes
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()

        x, y = event.xdata, event.ydata

        indx = searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_data( (minx, maxx), (y, y) )
        self.ly.set_data( (x, x), (miny, maxy) )

        self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
        print 'x=%1.2f, y=%1.2f'%(x,y)
        draw()

t = arange(0.0, 1.0, 0.01)
s = sin(2*2*pi*t)
ax = subplot(111)

cursor = Cursor(ax)
#cursor = SnaptoCursor(ax, t, s)
connect('motion_notify_event', cursor.mouse_move)

ax.plot(t, s, 'o')
axis([0,1,-1,1])
show()
