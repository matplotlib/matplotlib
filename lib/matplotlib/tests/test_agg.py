from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import io
import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from matplotlib.image import imread
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.testing.decorators import cleanup
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib import path


@cleanup
def test_repeated_save_with_alpha():
    # We want an image which has a background color of bluish green, with an
    # alpha of 0.25.

    fig = Figure([1, 0.4])
    canvas = FigureCanvas(fig)
    fig.set_facecolor((0, 1, 0.4))
    fig.patch.set_alpha(0.25)

    # The target color is fig.patch.get_facecolor()

    buf = io.BytesIO()

    fig.savefig(buf,
                facecolor=fig.get_facecolor(),
                edgecolor='none')

    # Save the figure again to check that the
    # colors don't bleed from the previous renderer.
    buf.seek(0)
    fig.savefig(buf,
                facecolor=fig.get_facecolor(),
                edgecolor='none')

    # Check the first pixel has the desired color & alpha
    # (approx: 0, 1.0, 0.4, 0.25)
    buf.seek(0)
    assert_array_almost_equal(tuple(imread(buf)[0, 0]),
                              (0.0, 1.0, 0.4, 0.250),
                              decimal=3)


@cleanup
def test_large_single_path_collection():
    buff = io.BytesIO()

    # Generates a too-large single path in a path collection that
    # would cause a segfault if the draw_markers optimization is
    # applied.
    f, ax = plt.subplots()
    collection = collections.PathCollection(
        [path.Path([[-10, 5], [10, 5], [10, -5], [-10, -5], [-10, 5]])])
    ax.add_artist(collection)
    ax.set_xlim(10**-3, 1)
    plt.savefig(buff)


def report_memory(i):
    pid = os.getpid()
    a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
    print(i, '  ', a2[1], end=' ')
    return int(a2[1].split()[0])

# This test is disabled -- it uses old API. -ADS 2009-09-07
## def test_memleak():
##     """Test agg backend for memory leaks."""
##     from matplotlib.ft2font import FT2Font
##     from numpy.random import rand
##     from matplotlib.backend_bases import GraphicsContextBase
##     from matplotlib.backends._backend_agg import RendererAgg

##     fontname = '/usr/local/share/matplotlib/Vera.ttf'

##     N = 200
##     for i in range( N ):
##         gc = GraphicsContextBase()
##         gc.set_clip_rectangle( [20, 20, 20, 20] )
##         o = RendererAgg( 400, 400, 72 )

##         for j in range( 50 ):
##             xs = [ 400*int(rand()) for k in range(8) ]
##             ys = [ 400*int(rand()) for k in range(8) ]
##             rgb = (1, 0, 0)
##             pnts = zip( xs, ys )
##             o.draw_polygon( gc, rgb, pnts )
##             o.draw_polygon( gc, None, pnts )

##         for j in range( 50 ):
##             x = [ 400*int(rand()) for k in range(4) ]
##             y = [ 400*int(rand()) for k in range(4) ]
##             o.draw_lines( gc, x, y )

##         for j in range( 50 ):
##             args = [ 400*int(rand()) for k in range(4) ]
##             rgb = (1, 0, 0)
##             o.draw_rectangle( gc, rgb, *args )

##         if 1: # add text
##             font = FT2Font( fontname )
##             font.clear()
##             font.set_text( 'hi mom', 60 )
##             font.set_size( 12, 72 )
##             o.draw_text_image( font.get_image(), 30, 40, gc )

##         fname = "agg_memleak_%05d.png"
##         o.write_png( fname % i )
##         val = report_memory( i )
##         if i==1: start = val

##     end = val
##     avgMem = (end - start) / float(N)
##     print 'Average memory consumed per loop: %1.4f\n' % (avgMem)

##     #TODO: Verify the expected mem usage and approximate tolerance that
##     # should be used
##     #self.checkClose( 0.32, avgMem, absTol = 0.1 )

##     # w/o text and w/o write_png: Average memory consumed per loop: 0.02
##     # w/o text and w/ write_png : Average memory consumed per loop: 0.3400
##     # w/ text and w/ write_png  : Average memory consumed per loop: 0.32


@cleanup
def test_marker_with_nan():
    # This creates a marker with nans in it, which was segfaulting the
    # Agg backend (see #3722)
    fig, ax = plt.subplots(1)
    steps = 1000
    data = np.arange(steps)
    ax.semilogx(data)
    ax.fill_between(data, data*0.8, data*1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
