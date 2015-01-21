#!/usr/bin/env pythonw
# Name: test_wxagg.py
# Purpose: exercises the agg to wx.Image and wx.Bitmap conversion functions
# Author: Ken McIvor <mcivor@iit.edu>
#
# Copyright 2005 Illinois Institute of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL ILLINOIS INSTITUTE OF TECHNOLOGY BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Except as contained in this notice, the name of Illinois Institute
# of Technology shall not be used in advertising or otherwise to promote
# the sale, use or other dealings in this Software without prior written
# authorization from Illinois Institute of Technology.

from __future__ import print_function


import wx
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox, Point, Value
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_wxagg import _py_convert_agg_to_wx_image, \
    _py_convert_agg_to_wx_bitmap
import matplotlib.backends._wxagg as wxagg


####################
# Test Configuration
####################

# Simple tests -- write PNG images of the plots
TEST_PY = 0
TEST_EXT = 0

# Timing tests -- print time per plot
TIME_PY = 1
TIME_EXT = 1


#################
# Test Parameters
#################

# Bounding box to use in testing
ll_x = 320
ll_y = 240
ur_x = 640
ur_y = 480
BBOX = Bbox(Point(Value(ll_x), Value(ll_y)),
            Point(Value(ur_x), Value(ur_y)))

# Number of iterations for timing
NITERS = 25


###############################################################################


#
# Testing framework
#

def time_loop(function, args):
    i = 0
    start = time.time()
    while i < NITERS:
        function(*args)
        i += 1
    return (time.time() - start) / NITERS


def make_figure():
    figure = Figure((6.4, 4.8), 100, frameon=False)
    canvas = FigureCanvasAgg(figure)
    return figure, canvas


def plot_sin(figure):
    from pylab import arange, sin, pi
    t = arange(0.0, 2.0, 0.01)
    s = sin(2 * pi * t)

    axes = figure.gca()
    axes.plot(t, s, linewidth=1.0)
    axes.set_title('title')


def main():
    app = wx.PySimpleApp()
    figure, canvas = make_figure()
    bbox = None
    plot_sin(figure)
    canvas.draw()
    agg = canvas.get_renderer()

    if 0:
        print('ll.x =', BBOX.ll().x().get())
        print('ll.y =', BBOX.ll().y().get())
        print('ur.x =', BBOX.ur().x().get())
        print('ur.y =', BBOX.ur().y().get())

    # test the pure python implementation
    if TEST_PY:
        i_py = _py_convert_agg_to_wx_image(agg, None)
        b_py = _py_convert_agg_to_wx_bitmap(agg, None)
        i_py_b = _py_convert_agg_to_wx_image(agg, BBOX)
        b_py_b = _py_convert_agg_to_wx_bitmap(agg, BBOX)

        i_py.SaveFile('a_py_img.png', wx.BITMAP_TYPE_PNG)
        b_py.SaveFile('a_py_bmp.png', wx.BITMAP_TYPE_PNG)
        i_py_b.SaveFile('b_py_img.png', wx.BITMAP_TYPE_PNG)
        b_py_b.SaveFile('b_py_bmp.png', wx.BITMAP_TYPE_PNG)

    # test the C++ implementation
    if TEST_EXT:
        i_ext = wxagg.convert_agg_to_wx_image(agg, None)
        b_ext = wxagg.convert_agg_to_wx_bitmap(agg, None)
        i_ext_b = wxagg.convert_agg_to_wx_image(agg, BBOX)
        b_ext_b = wxagg.convert_agg_to_wx_bitmap(agg, BBOX)

        i_ext.SaveFile('a_ext_img.png', wx.BITMAP_TYPE_PNG)
        b_ext.SaveFile('a_ext_bmp.png', wx.BITMAP_TYPE_PNG)
        i_ext_b.SaveFile('b_ext_img.png', wx.BITMAP_TYPE_PNG)
        b_ext_b.SaveFile('b_ext_bmp.png', wx.BITMAP_TYPE_PNG)

    # time the pure python implementation
    if TIME_PY:
        t = time_loop(_py_convert_agg_to_wx_image, (agg, None))
        print('Python agg2img:        %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(_py_convert_agg_to_wx_bitmap, (agg, None))
        print('Python agg2bmp:        %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(_py_convert_agg_to_wx_image, (agg, BBOX))
        print('Python agg2img w/bbox: %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(_py_convert_agg_to_wx_bitmap, (agg, BBOX))
        print('Python agg2bmp w/bbox: %.4f seconds (%.1f HZ)' % (t, 1 / t))

    # time the C++ implementation
    if TIME_EXT:
        t = time_loop(wxagg.convert_agg_to_wx_image, (agg, None))
        print('_wxagg agg2img:        %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(wxagg.convert_agg_to_wx_bitmap, (agg, None))
        print('_wxagg agg2bmp:        %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(wxagg.convert_agg_to_wx_image, (agg, BBOX))
        print('_wxagg agg2img w/bbox: %.4f seconds (%.1f HZ)' % (t, 1 / t))

        t = time_loop(wxagg.convert_agg_to_wx_bitmap, (agg, BBOX))
        print('_wxagg agg2bmp w/bbox: %.4f seconds (%.1f HZ)' % (t, 1 / t))


if __name__ == '__main__':
    main()
