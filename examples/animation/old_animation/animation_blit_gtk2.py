#!/usr/bin/env python

from __future__ import print_function

"""
This example utlizes restore_region with optional bbox and xy
arguments.  The plot is continuously shifted to the left. Instead of
drawing everything again, the plot is saved (copy_from_bbox) and
restored with offset by the amount of the shift. And only newly
exposed area is drawn. This technique may reduce drawing time for some cases.
"""

import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt

class UpdateLine(object):
    def get_bg_bbox(self):
        
        return self.ax.bbox.padded(-3)
    
    def __init__(self, canvas, ax):
        self.cnt = 0
        self.canvas = canvas
        self.ax = ax

        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_pixel_offset = 0.
        

        self.x0 = 0
        self.phases = np.random.random_sample((20,)) * np.pi * 2
        self.line, = ax.plot([], [], "-", animated=True, lw=2)

        self.point, = ax.plot([], [], "ro", animated=True, lw=2)

        self.ax.set_ylim(-1.1, 1.1)

        self.background1 = None

        cmap = plt.cm.jet
        from itertools import cycle
        self.color_cycle = cycle(cmap(np.arange(cmap.N)))


    def save_bg(self):
        self.background1 = self.canvas.copy_from_bbox(self.ax.get_figure().bbox)

        self.background2 = self.canvas.copy_from_bbox(self.get_bg_bbox())


    def get_dx_data(self, dx_pixel):
        tp = self.ax.transData.inverted().transform_point
        x0, y0 = tp((0, 0))
        x1, y1 = tp((dx_pixel, 0))
        return (x1-x0)


    def restore_background_shifted(self, dx_pixel):
        """
        restore bacground shifted by dx in data coordinate. This only
        works if the data coordinate system is linear.
        """

        # restore the clean slate background
        self.canvas.restore_region(self.background1)

        # restore subregion (x1+dx, y1, x2, y2) of the second bg 
        # in a offset position (x1-dx, y1)
        x1, y1, x2, y2 = self.background2.get_extents()
        self.canvas.restore_region(self.background2,
                                   bbox=(x1+dx_pixel, y1, x2, y2),
                                   xy=(x1-dx_pixel, y1))

        return dx_pixel

    def on_draw(self, *args):
        self.save_bg()
        return False
    
    def update_line(self, *args):

        if self.background1 is None:
            return True
        
        cur_time = time.time()
        pixel_offset = int((cur_time - self.start_time)*100.)
        dx_pixel = pixel_offset - self.prev_pixel_offset
        self.prev_pixel_offset = pixel_offset
        dx_data = self.get_dx_data(dx_pixel) #cur_time - self.prev_time)
        
        x0 = self.x0
        self.x0 += dx_data
        self.prev_time = cur_time

        self.ax.set_xlim(self.x0-2, self.x0+0.1)


        # restore background which will plot lines from previous plots
        self.restore_background_shifted(dx_pixel) #x0, self.x0)
        # This restores lines between [x0-2, x0]



        self.line.set_color(self.color_cycle.next())

        # now plot line segment within [x0, x0+dx_data], 
        # Note that we're only plotting a line between [x0, x0+dx_data].
        xx = np.array([x0, self.x0])
        self.line.set_xdata(xx)

        # the for loop below could be improved by using collection.
        [(self.line.set_ydata(np.sin(xx+p)),
          self.ax.draw_artist(self.line)) \
         for p in self.phases]

        self.background2 = canvas.copy_from_bbox(self.get_bg_bbox())

        self.point.set_xdata([self.x0])

        [(self.point.set_ydata(np.sin([self.x0+p])),
          self.ax.draw_artist(self.point)) \
         for p in self.phases]


        self.ax.draw_artist(self.ax.xaxis)
        self.ax.draw_artist(self.ax.yaxis)

        self.canvas.blit(self.ax.get_figure().bbox)


        dt = (time.time()-tstart)
        if dt>15:
            # print the timing info and quit
            print('FPS:' , self.cnt/dt)
            gtk.main_quit()
            raise SystemExit

        self.cnt += 1
        return True


plt.rcParams["text.usetex"] = False

fig, ax = plt.subplots()
ax.xaxis.set_animated(True)
ax.yaxis.set_animated(True)
canvas = fig.canvas

fig.subplots_adjust(left=0.2, bottom=0.2)
canvas.draw()

# for profiling
tstart = time.time()

ul = UpdateLine(canvas, ax)
gobject.idle_add(ul.update_line)

canvas.mpl_connect('draw_event', ul.on_draw)

plt.show()
