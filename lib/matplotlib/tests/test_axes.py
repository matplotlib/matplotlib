import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
import pylab

@image_comparison(baseline_images=['formatter_ticker_001',
                                   'formatter_ticker_002',
                                   'formatter_ticker_003',
                                   'formatter_ticker_004',
                                   'formatter_ticker_005',
                                   ])
def test_formatter_ticker():
    """Test Some formatter and ticker issues."""
    import matplotlib.testing.jpl_units as units
    units.register()

    # This essentially test to see if user specified labels get overwritten
    # by the auto labeler functionality of the axes.
    xdata = [ x*units.sec for x in range(10) ]
    ydata1 = [ (1.5*y - 0.5)*units.km for y in range(10) ]
    ydata2 = [ (1.75*y - 1.0)*units.km for y in range(10) ]

    fig = pylab.figure()
    ax = pylab.subplot( 111 )
    ax.set_xlabel( "x-label 001" )
    fig.savefig( 'formatter_ticker_001' )

    ax.plot( xdata, ydata1, color='blue', xunits="sec" )
    fig.savefig( 'formatter_ticker_002' )

    ax.set_xlabel( "x-label 003" )
    fig.savefig( 'formatter_ticker_003' )

    ax.plot( xdata, ydata2, color='green', xunits="hour" )
    ax.set_xlabel( "x-label 004" )
    fig.savefig( 'formatter_ticker_004' )

    # See SF bug 2846058
    # https://sourceforge.net/tracker/?func=detail&aid=2846058&group_id=80706&atid=560720
    ax.set_xlabel( "x-label 005" )
    ax.autoscale_view()
    fig.savefig( 'formatter_ticker_005' )

@image_comparison(baseline_images=['offset_points'])
def test_basic_annotate():
    # Setup some data
    t = np.arange( 0.0, 5.0, 0.01 )
    s = np.cos( 2.0*np.pi * t )

    # Offset Points

    fig = pylab.figure()
    ax = fig.add_subplot( 111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5) )
    line, = ax.plot( t, s, lw=3, color='purple' )

    ax.annotate( 'local max', xy=(3, 1), xycoords='data',
                 xytext=(3, 3), textcoords='offset points' )

    fig.savefig( 'offset_points' )

@image_comparison(baseline_images=['polar_axes'])
def test_polar_annotations():
    """Polar Plot Annotations"""

    # you can specify the xypoint and the xytext in different
    # positions and coordinate systems, and optionally turn on a
    # connecting line and mark the point with a marker.  Annotations
    # work on polar axes too.  In the example below, the xy point is
    # in native coordinates (xycoords defaults to 'data').  For a
    # polar axes, this is in (theta, radius) space.  The text in this
    # example is placed in the fractional figure coordinate system.
    # Text keyword args like horizontal and vertical alignment are
    # respected

    # Setup some data
    r = np.arange(0.0, 1.0, 0.001 )
    theta = 2.0 * 2.0 * np.pi * r

    fig = pylab.figure()
    ax = fig.add_subplot( 111, polar=True )
    line, = ax.plot( theta, r, color='#ee8d18', lw=3 )

    ind = 800
    thisr, thistheta = r[ind], theta[ind]
    ax.plot([thistheta], [thisr], 'o')
    ax.annotate('a polar annotation',
                xy=(thistheta, thisr),  # theta, radius
                xytext=(0.05, 0.05),    # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom',
                )

    fig.savefig( 'polar_axes' )

   #--------------------------------------------------------------------
@image_comparison(baseline_images=['polar_coords'])
def test_polar_coord_annotations():
    """Polar Coordinate Annotations"""

    # You can also use polar notation on a catesian axes.  Here the
    # native coordinate system ('data') is cartesian, so you need to
    # specify the xycoords and textcoords as 'polar' if you want to
    # use (theta, radius)
    from matplotlib.patches import Ellipse
    el = Ellipse((0,0), 10, 20, facecolor='r', alpha=0.5)

    fig = pylab.figure()
    ax = fig.add_subplot( 111, aspect='equal' )

    ax.add_artist( el )
    el.set_clip_box( ax.bbox )

    ax.annotate('the top',
                xy=(np.pi/2., 10.),      # theta, radius
                xytext=(np.pi/3, 20.),   # theta, radius
                xycoords='polar',
                textcoords='polar',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom',
                clip_on=True, # clip to the axes bounding box
                )

    ax.set_xlim( -20, 20 )
    ax.set_ylim( -20, 20 )
    fig.savefig( 'polar_coords' )

@image_comparison(baseline_images=['fill_units'])
def test_fill_units():
    """Test the fill method with unitized-data."""
    from datetime import datetime
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t = units.Epoch( "ET", dt=datetime(2009, 4, 27) )
    value = 10.0 * units.deg
    day = units.Duration( "ET", 24.0 * 60.0 * 60.0 )

    fig = pylab.figure()

    # Top-Left
    ax1 = fig.add_subplot( 221 )
    ax1.plot( [t], [value], yunits='deg', color='red' )
    ax1.fill( [733525.0, 733525.0, 733526.0, 733526.0],
              [0.0, 0.0, 90.0, 0.0], 'b' )

    # Top-Right
    ax2 = fig.add_subplot( 222 )
    ax2.plot( [t], [value], yunits='deg', color='red' )
    ax2.fill( [t,      t,      t+day,     t+day],
              [0.0,  0.0,  90.0,    0.0], 'b' )

    # Bottom-Left
    ax3 = fig.add_subplot( 223 )
    ax3.plot( [t], [value], yunits='deg', color='red' )
    ax1.fill( [733525.0, 733525.0, 733526.0, 733526.0],
              [0*units.deg,  0*units.deg,  90*units.deg,    0*units.deg], 'b' )

    # Bottom-Right
    ax4 = fig.add_subplot( 224 )
    ax4.plot( [t], [value], yunits='deg', color='red' )
    ax4.fill( [t,      t,      t+day,     t+day],
              [0*units.deg,  0*units.deg,  90*units.deg,    0*units.deg],
              facecolor="blue" )

    fig.autofmt_xdate()
    fig.savefig( 'fill_units' )

@image_comparison(baseline_images=['single_point'])
def test_single_point():
    """Test single-point plots."""
    fig = pylab.figure()
    pylab.subplot( 211 )
    pylab.plot( [0], [0], 'o' )

    pylab.subplot( 212 )
    pylab.plot( [1], [1], 'o' )

    fig.savefig( 'single_point' )
