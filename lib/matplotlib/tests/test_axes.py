from nose.tools import assert_equal
from nose.tools import assert_raises

import numpy as np
from numpy import ma

import matplotlib
from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['formatter_ticker_001',
                                   'formatter_ticker_002',
                                   'formatter_ticker_003',
                                   'formatter_ticker_004',
                                   'formatter_ticker_005',
                                   ])
def test_formatter_ticker():
    import matplotlib.testing.jpl_units as units
    units.register()

    # This should affect the tick size.  (Tests issue #543)
    matplotlib.rcParams['lines.markeredgewidth'] = 30

    # This essentially test to see if user specified labels get overwritten
    # by the auto labeler functionality of the axes.
    xdata = [ x*units.sec for x in range(10) ]
    ydata1 = [ (1.5*y - 0.5)*units.km for y in range(10) ]
    ydata2 = [ (1.75*y - 1.0)*units.km for y in range(10) ]

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.set_xlabel( "x-label 001" )

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.set_xlabel( "x-label 001" )
    ax.plot( xdata, ydata1, color='blue', xunits="sec" )

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.set_xlabel( "x-label 001" )
    ax.plot( xdata, ydata1, color='blue', xunits="sec" )
    ax.set_xlabel( "x-label 003" )

    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.plot( xdata, ydata1, color='blue', xunits="sec" )
    ax.plot( xdata, ydata2, color='green', xunits="hour" )
    ax.set_xlabel( "x-label 004" )

    # See SF bug 2846058
    # https://sourceforge.net/tracker/?func=detail&aid=2846058&group_id=80706&atid=560720
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.plot( xdata, ydata1, color='blue', xunits="sec" )
    ax.plot( xdata, ydata2, color='green', xunits="hour" )
    ax.set_xlabel( "x-label 005" )
    ax.autoscale_view()

@cleanup
def test_add_collection():
    # Test if data limits are unchanged by adding an empty collection.
    # Github issue #1490, pull #1497.
    fig = matplotlib.figure.Figure()
    fig2 = matplotlib.figure.Figure()
    ax = fig.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    coll = ax2.scatter([0, 1], [0, 1])
    ax.add_collection(coll)
    bounds = ax.dataLim.bounds
    coll = ax2.scatter([], [])
    ax.add_collection(coll)
    assert ax.dataLim.bounds == bounds

@image_comparison(baseline_images=["formatter_large_small"])
def test_formatter_large_small():
    # github issue #617, pull #619
    fig, ax = plt.subplots(1)
    x = [0.500000001, 0.500000002]
    y = [1e64, 1.1e64]
    ax.plot(x, y)

@image_comparison(baseline_images=["twin_axis_locaters_formatters"])
def test_twin_axis_locaters_formatters():
    vals = np.linspace(0, 1, num=5, endpoint=True)
    locs = np.sin(np.pi * vals / 2.0)

    majl = plt.FixedLocator(locs)
    minl = plt.FixedLocator([0.1, 0.2, 0.3])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot([0.1, 100], [0, 1])
    ax1.yaxis.set_major_locator(majl)
    ax1.yaxis.set_minor_locator(minl)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%08.2lf'))
    ax1.yaxis.set_minor_formatter(plt.FixedFormatter(['tricks', 'mind', 'jedi']))

    ax1.xaxis.set_major_locator(plt.LinearLocator())
    ax1.xaxis.set_minor_locator(plt.FixedLocator([15, 35, 55, 75]))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%05.2lf'))
    ax1.xaxis.set_minor_formatter(plt.FixedFormatter(['c', '3', 'p', 'o']))
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()

@image_comparison(baseline_images=["autoscale_tiny_range"], remove_text=True)
def test_autoscale_tiny_range():
    # github pull #904
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    for i in xrange(4):
        y1 = 10**(-11 - i)
        ax[i].plot([0, 1], [1, 1 + y1])

@image_comparison(baseline_images=['offset_points'],
                  remove_text=True)
def test_basic_annotate():
    # Setup some data
    t = np.arange( 0.0, 5.0, 0.01 )
    s = np.cos( 2.0*np.pi * t )

    # Offset Points

    fig = plt.figure()
    ax = fig.add_subplot( 111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5) )
    line, = ax.plot( t, s, lw=3, color='purple' )

    ax.annotate( 'local max', xy=(3, 1), xycoords='data',
                 xytext=(3, 3), textcoords='offset points' )

@image_comparison(baseline_images=['polar_axes'])
def test_polar_annotations():
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

    fig = plt.figure()
    ax = fig.add_subplot( 111, polar=True )
    line, = ax.plot( theta, r, color='#ee8d18', lw=3 )
    line, = ax.plot( (0, 0), (0, 1), color="#0000ff", lw=1)

    ind = 800
    thisr, thistheta = r[ind], theta[ind]
    ax.plot([thistheta], [thisr], 'o')
    ax.annotate('a polar annotation',
                xy=(thistheta, thisr),  # theta, radius
                xytext=(0.05, 0.05),    # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='baseline',
                )

   #--------------------------------------------------------------------
@image_comparison(baseline_images=['polar_coords'],
                  remove_text=True)
def test_polar_coord_annotations():
    # You can also use polar notation on a catesian axes.  Here the
    # native coordinate system ('data') is cartesian, so you need to
    # specify the xycoords and textcoords as 'polar' if you want to
    # use (theta, radius)
    from matplotlib.patches import Ellipse
    el = Ellipse((0,0), 10, 20, facecolor='r', alpha=0.5)

    fig = plt.figure()
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
                verticalalignment='baseline',
                clip_on=True, # clip to the axes bounding box
                )

    ax.set_xlim( -20, 20 )
    ax.set_ylim( -20, 20 )

@image_comparison(baseline_images=['fill_units'], tol=18, extensions=['png'],
                  savefig_kwarg={'dpi': 60})
def test_fill_units():
    from datetime import datetime
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t = units.Epoch( "ET", dt=datetime(2009, 4, 27) )
    value = 10.0 * units.deg
    day = units.Duration( "ET", 24.0 * 60.0 * 60.0 )

    fig = plt.figure()

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
    ax3.fill( [733525.0, 733525.0, 733526.0, 733526.0],
              [0*units.deg,  0*units.deg,  90*units.deg,    0*units.deg], 'b' )

    # Bottom-Right
    ax4 = fig.add_subplot( 224 )
    ax4.plot( [t], [value], yunits='deg', color='red' )
    ax4.fill( [t,      t,      t+day,     t+day],
              [0*units.deg,  0*units.deg,  90*units.deg,    0*units.deg],
              facecolor="blue" )

    fig.autofmt_xdate()

@image_comparison(baseline_images=['single_point'])
def test_single_point():
    # Issue #1796: don't let lines.marker affect the grid
    matplotlib.rcParams['lines.marker'] = 'o'
    matplotlib.rcParams['axes.grid'] = True

    fig = plt.figure()
    plt.subplot( 211 )
    plt.plot( [0], [0], 'o' )

    plt.subplot( 212 )
    plt.plot( [1], [1], 'o' )

@image_comparison(baseline_images=['single_date'])
def test_single_date():
    time1=[ 721964.0 ]
    data1=[ -65.54 ]

    fig = plt.figure()
    plt.subplot( 211 )
    plt.plot_date( time1, data1, 'o', color='r' )

    plt.subplot( 212 )
    plt.plot( time1, data1, 'o', color='r' )

@image_comparison(baseline_images=['shaped_data'])
def test_shaped_data():
    xdata = np.array([[ 0.53295185,  0.23052951,  0.19057629,  0.66724975,  0.96577916,
                        0.73136095,  0.60823287,  0.017921  ,  0.29744742,  0.27164665],
                      [ 0.2798012 ,  0.25814229,  0.02818193,  0.12966456,  0.57446277,
                        0.58167607,  0.71028245,  0.69112737,  0.89923072,  0.99072476],
                      [ 0.81218578,  0.80464528,  0.76071809,  0.85616314,  0.12757994,
                        0.94324936,  0.73078663,  0.09658102,  0.60703967,  0.77664978],
                      [ 0.28332265,  0.81479711,  0.86985333,  0.43797066,  0.32540082,
                        0.43819229,  0.92230363,  0.49414252,  0.68168256,  0.05922372],
                      [ 0.10721335,  0.93904142,  0.79163075,  0.73232848,  0.90283839,
                        0.68408046,  0.25502302,  0.95976614,  0.59214115,  0.13663711],
                      [ 0.28087456,  0.33127607,  0.15530412,  0.76558121,  0.83389773,
                        0.03735974,  0.98717738,  0.71432229,  0.54881366,  0.86893953],
                      [ 0.77995937,  0.995556  ,  0.29688434,  0.15646162,  0.051848  ,
                        0.37161935,  0.12998491,  0.09377296,  0.36882507,  0.36583435],
                      [ 0.37851836,  0.05315792,  0.63144617,  0.25003433,  0.69586032,
                        0.11393988,  0.92362096,  0.88045438,  0.93530252,  0.68275072],
                      [ 0.86486596,  0.83236675,  0.82960664,  0.5779663 ,  0.25724233,
                        0.84841095,  0.90862812,  0.64414887,  0.3565272 ,  0.71026066],
                      [ 0.01383268,  0.3406093 ,  0.76084285,  0.70800694,  0.87634056,
                        0.08213693,  0.54655021,  0.98123181,  0.44080053,  0.86815815]])

    y1 = np.arange( 10 )
    y1.shape = 1, 10

    y2 = np.arange( 10 )
    y2.shape = 10, 1

    fig = plt.figure()
    plt.subplot( 411 )
    plt.plot( y1 )
    plt.subplot( 412 )
    plt.plot( y2 )

    plt.subplot( 413 )
    assert_raises(ValueError,plt.plot, (y1,y2))

    plt.subplot( 414 )
    plt.plot( xdata[:,1], xdata[1,:], 'o' )

@image_comparison(baseline_images=['const_xy'])
def test_const_xy():
    fig = plt.figure()

    plt.subplot( 311 )
    plt.plot( np.arange(10), np.ones( (10,) ) )

    plt.subplot( 312 )
    plt.plot( np.ones( (10,) ), np.arange(10) )

    plt.subplot( 313 )
    plt.plot( np.ones( (10,) ), np.ones( (10,) ), 'o' )


@image_comparison(baseline_images=['polar_wrap_180',
                                   'polar_wrap_360',
                                   ])
def test_polar_wrap():
    D2R = np.pi / 180.0

    fig = plt.figure()

    plt.subplot(111, polar=True)

    plt.polar( [179*D2R, -179*D2R], [0.2, 0.1], "b.-" )
    plt.polar( [179*D2R,  181*D2R], [0.2, 0.1], "g.-" )
    plt.rgrids( [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] )
    assert len(fig.axes) == 1, 'More than one polar axes created.'
    fig = plt.figure()

    plt.subplot( 111, polar=True)
    plt.polar( [2*D2R, -2*D2R], [0.2, 0.1], "b.-" )
    plt.polar( [2*D2R,  358*D2R], [0.2, 0.1], "g.-" )
    plt.polar( [358*D2R,  2*D2R], [0.2, 0.1], "r.-" )
    plt.rgrids( [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] )


@image_comparison(baseline_images=['polar_units', 'polar_units_2'])
def test_polar_units():
    import matplotlib.testing.jpl_units as units
    from nose.tools import assert_true
    units.register()

    pi = np.pi
    deg = units.UnitDbl( 1.0, "deg" )
    km = units.UnitDbl( 1.0, "km" )

    x1 = [ pi/6.0, pi/4.0, pi/3.0, pi/2.0 ]
    x2 = [ 30.0*deg, 45.0*deg, 60.0*deg, 90.0*deg ]

    y1 = [ 1.0, 2.0, 3.0, 4.0]
    y2 = [ 4.0, 3.0, 2.0, 1.0 ]

    fig = plt.figure()

    plt.polar( x2, y1, color = "blue" )

    # polar( x2, y1, color = "red", xunits="rad" )
    # polar( x2, y2, color = "green" )

    fig = plt.figure()

    # make sure runits and theta units work
    y1 = [ y*km for y in y1 ]
    plt.polar( x2, y1, color = "blue", thetaunits="rad", runits="km" )
    assert_true( isinstance(plt.gca().get_xaxis().get_major_formatter(), units.UnitDblFormatter) )


@image_comparison(baseline_images=['polar_rmin'])
def test_polar_rmin():
    r = np.arange(0, 3.0, 0.01)
    theta = 2*np.pi*r

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_rmax(2.0)
    ax.set_rmin(0.5)

@image_comparison(baseline_images=['polar_theta_position'])
def test_polar_theta_position():
    r = np.arange(0, 3.0, 0.01)
    theta = 2*np.pi*r

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_theta_zero_location("NW")
    ax.set_theta_direction('clockwise')

@image_comparison(baseline_images=['axvspan_epoch'])
def test_axvspan_epoch():
    from datetime import datetime
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t0 = units.Epoch( "ET", dt=datetime(2009, 1, 20) )
    tf = units.Epoch( "ET", dt=datetime(2009, 1, 21) )

    dt = units.Duration( "ET", units.day.convert( "sec" ) )

    fig = plt.figure()

    plt.axvspan( t0, tf, facecolor="blue", alpha=0.25 )

    ax = plt.gca()
    ax.set_xlim( t0 - 5.0*dt, tf + 5.0*dt )

@image_comparison(baseline_images=['axhspan_epoch'])
def test_axhspan_epoch():
    from datetime import datetime
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t0 = units.Epoch( "ET", dt=datetime(2009, 1, 20) )
    tf = units.Epoch( "ET", dt=datetime(2009, 1, 21) )

    dt = units.Duration( "ET", units.day.convert( "sec" ) )

    fig = plt.figure()

    plt.axhspan( t0, tf, facecolor="blue", alpha=0.25 )

    ax = plt.gca()
    ax.set_ylim( t0 - 5.0*dt, tf + 5.0*dt )


@image_comparison(baseline_images=['hexbin_extent'],
                  remove_text=True, extensions=['png'])
def test_hexbin_extent():
    # this test exposes sf bug 2856228
    fig = plt.figure()

    ax = fig.add_subplot(111)
    data = np.arange(2000.)/2000.
    data.shape = 2, 1000
    x, y = data

    ax.hexbin(x, y, extent=[.1, .3, .6, .7])

@cleanup
def test_hexbin_pickable():
    # From #1973: Test that picking a hexbin collection works
    class FauxMouseEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    fig = plt.figure()

    ax = fig.add_subplot(111)
    data = np.arange(200.)/200.
    data.shape = 2, 100
    x, y = data
    hb = ax.hexbin(x, y, extent=[.1, .3, .6, .7], picker=1)

    assert hb.contains(FauxMouseEvent(400, 300))[0]

@image_comparison(baseline_images=['hexbin_log'],
                  remove_text=True,
                  extensions=['png'])
def test_hexbin_log():
    # Issue #1636
    fig = plt.figure()

    np.random.seed(0)
    n = 100000
    x = np.random.standard_normal(n)
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
    y = np.power(2, y * 0.5)
    ax = fig.add_subplot(111)
    ax.hexbin(x, y, yscale='log')

@cleanup
def test_inverted_limits():
    # Test gh:1553
    # Calling invert_xaxis prior to plotting should not disable autoscaling
    # while still maintaining the inverted direction
    fig = plt.figure()
    ax = fig.gca()
    ax.invert_xaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    assert ax.get_xlim() == (4, -5)
    assert ax.get_ylim() == (-3, 5)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    ax.invert_yaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    assert ax.get_xlim() == (-5, 4)
    assert ax.get_ylim() == (5, -3)
    plt.close()

@image_comparison(baseline_images=['nonfinite_limits'])
def test_nonfinite_limits():
    x = np.arange(0., np.e, 0.01)
    olderr = np.seterr(divide='ignore') #silence divide by zero warning from log(0)
    try:
        y = np.log(x)
    finally:
        np.seterr(**olderr)
    x[len(x)/2] = np.nan
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)

@image_comparison(baseline_images=['imshow'],
                  remove_text=True)
def test_imshow():
    #Create a NxN image
    N=100
    (x,y) = np.indices((N,N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    #Create a contour plot at N/4 and extract both the clip path and transform
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(r)

@image_comparison(baseline_images=['imshow_clip'])
def test_imshow_clip():
    # As originally reported by Gellule Xg <gellule.xg@free.fr>

    #Create a NxN image
    N=100
    (x,y) = np.indices((N,N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    #Create a contour plot at N/4 and extract both the clip path and transform
    fig = plt.figure()
    ax = fig.add_subplot(111)

    c = ax.contour(r,[N/4])
    x = c.collections[0]
    clipPath = x.get_paths()[0]
    clipTransform = x.get_transform()

    from matplotlib.transforms import TransformedPath
    clip_path = TransformedPath(clipPath, clipTransform)

    #Plot the image clipped by the contour
    ax.imshow(r, clip_path=clip_path)

@image_comparison(baseline_images=['polycollection_joinstyle'],
                  remove_text=True)
def test_polycollection_joinstyle():
    # Bug #2890979 reported by Matthew West

    from matplotlib import collections as mcoll

    fig = plt.figure()
    ax = fig.add_subplot(111)
    verts = np.array([[1,1], [1,2], [2,2], [2,1]])
    c = mcoll.PolyCollection([verts], linewidths = 40)
    ax.add_collection(c)
    ax.set_xbound(0, 3)
    ax.set_ybound(0, 3)

@image_comparison(baseline_images=['fill_between_interpolate'],
                  remove_text=True)
def test_fill_between_interpolate():
    x = np.arange(0.0, 2, 0.02)
    y1 = np.sin(2*np.pi*x)
    y2 = 1.2*np.sin(4*np.pi*x)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(x, y1, x, y2, color='black')
    ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='white', hatch='/', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2<=y1, facecolor='red', interpolate=True)

    # Test support for masked arrays.
    y2 = np.ma.masked_greater(y2, 1.0)
    # Test that plotting works for masked arrays with the first element masked
    y2[0] = np.ma.masked
    ax1 = fig.add_subplot(212, sharex=ax)
    ax1.plot(x, y1, x, y2, color='black')
    ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='green', interpolate=True)
    ax1.fill_between(x, y1, y2, where=y2<=y1, facecolor='red', interpolate=True)

@image_comparison(baseline_images=['symlog'])
def test_symlog():
    x = np.array([0,1,2,4,6,9,12,24])
    y = np.array([1000000, 500000, 100000, 100, 5, 0, 0, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_yscale('symlog')
    ax.set_xscale=('linear')
    ax.set_ylim(-1,10000000)

@image_comparison(baseline_images=['symlog2'],
                  remove_text=True)
def test_symlog2():
    # Numbers from -50 to 50, with 0.1 as step
    x = np.arange(-50,50, 0.001)

    fig = plt.figure()
    ax = fig.add_subplot(511)
    # Plots a simple linear function 'f(x) = x'
    ax.plot(x, x)
    ax.set_xscale('symlog', linthreshx=20.0)
    ax.grid(True)

    ax = fig.add_subplot(512)
    # Plots a simple linear function 'f(x) = x'
    ax.plot(x, x)
    ax.set_xscale('symlog', linthreshx=2.0)
    ax.grid(True)

    ax = fig.add_subplot(513)
    # Plots a simple linear function 'f(x) = x'
    ax.plot(x, x)
    ax.set_xscale('symlog', linthreshx=1.0)
    ax.grid(True)

    ax = fig.add_subplot(514)
    # Plots a simple linear function 'f(x) = x'
    ax.plot(x, x)
    ax.set_xscale('symlog', linthreshx=0.1)
    ax.grid(True)

    ax = fig.add_subplot(515)
    # Plots a simple linear function 'f(x) = x'
    ax.plot(x, x)
    ax.set_xscale('symlog', linthreshx=0.01)
    ax.grid(True)
    ax.set_ylim(-0.1, 0.1)

@image_comparison(baseline_images=['pcolormesh'], remove_text=True)
def test_pcolormesh():
    n = 12
    x = np.linspace(-1.5,1.5,n)
    y = np.linspace(-1.5,1.5,n*2)
    X,Y = np.meshgrid(x,y);
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Qx = (Qx + 1.1)
    Z = np.sqrt(X**2 + Y**2)/5;
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # The color array can include masked values:
    Zm = ma.masked_where(np.fabs(Qz) < 0.5*np.amax(Qz), Z)

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.pcolormesh(Qx,Qz,Z, lw=0.5, edgecolors='k')

    ax = fig.add_subplot(132)
    ax.pcolormesh(Qx,Qz,Z, lw=2, edgecolors=['b', 'w'])

    ax = fig.add_subplot(133)
    ax.pcolormesh(Qx,Qz,Z, shading="gouraud")

def test_pcolorargs():
    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)/5

    _, ax = plt.subplots()
    assert_raises(TypeError, ax.pcolormesh, y, x, Z)
    assert_raises(TypeError, ax.pcolormesh, X, Y, Z.T)
    assert_raises(TypeError, ax.pcolormesh, x, y, Z[:-1, :-1],
                  shading="gouraud")
    assert_raises(TypeError, ax.pcolormesh, X, Y, Z[:-1, :-1],
                  shading="gouraud")

@image_comparison(baseline_images=['canonical'])
def test_canonical():
    fig, ax = plt.subplots()
    ax.plot([1,2,3])


@image_comparison(baseline_images=['arc_ellipse'],
                  remove_text=True)
def test_arc_ellipse():
    from matplotlib import patches
    xcenter, ycenter = 0.38, 0.52
    width, height = 1e-1, 3e-1
    angle = -30

    theta = np.arange(0.0, 360.0, 1.0)*np.pi/180.0
    x = width/2. * np.cos(theta)
    y = height/2. * np.sin(theta)

    rtheta = angle*np.pi/180.
    R = np.array([
        [np.cos(rtheta),  -np.sin(rtheta)],
        [np.sin(rtheta), np.cos(rtheta)],
        ])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter

    fig = plt.figure()
    ax = fig.add_subplot(211, aspect='auto')
    ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)

    e1 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e1)

    ax = fig.add_subplot(212, aspect='equal')
    ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
    e2 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e2)

@image_comparison(baseline_images=['units_strings'])
def test_units_strings():
    # Make sure passing in sequences of strings doesn't cause the unit
    # conversion registry to recurse infinitely
    Id = ['50', '100', '150', '200', '250']
    pout = ['0', '7.4', '11.4', '14.2', '16.3']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Id, pout)

@image_comparison(baseline_images=['markevery'],
                  remove_text=True)
def test_markevery():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # check marker only plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o', label='default')
    ax.plot(x, y, 'd', markevery=None, label='mark all')
    ax.plot(x, y, 's', markevery=10, label='mark every 10')
    ax.plot(x, y, '+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()

@image_comparison(baseline_images=['markevery_line'],
                  remove_text=True)
def test_markevery_line():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # check line/marker combos
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-o', label='default')
    ax.plot(x, y, '-d', markevery=None, label='mark all')
    ax.plot(x, y, '-s', markevery=10, label='mark every 10')
    ax.plot(x, y, '-+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()

@image_comparison(baseline_images=['marker_edges'],
                  remove_text=True)
def test_marker_edges():
    x = np.linspace(0, 1, 10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, np.sin(x), 'y.', ms=30.0, mew=0, mec='r')
    ax.plot(x+0.1, np.sin(x), 'y.', ms=30.0, mew=1, mec='r')
    ax.plot(x+0.2, np.sin(x), 'y.', ms=30.0, mew=2, mec='b')

@image_comparison(baseline_images=['hist_log'],
                  remove_text=True)
def test_hist_log():
    data0 = np.linspace(0,1,200)**3
    data = np.r_[1-data0, 1+data0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, fill=False, log=True)

@image_comparison(baseline_images=['hist_steplog'], remove_text=True)
def test_hist_steplog():
    np.random.seed(0)
    data = np.random.standard_normal(2000)
    data += -2.0 - np.min(data)
    data_pos = data + 2.1
    data_big = data_pos + 30

    ax = plt.subplot(3, 1, 1)
    plt.hist(data, 100, histtype='stepfilled', log=True)

    ax = plt.subplot(3, 1, 2)
    plt.hist(data_pos, 100, histtype='stepfilled', log=True)

    ax = plt.subplot(3, 1, 3)
    plt.hist(data_big, 100, histtype='stepfilled', log=True, orientation='horizontal')

def contour_dat():
    x = np.linspace(-3, 5, 150)
    y = np.linspace(-3, 5, 120)
    z = np.cos(x) + np.sin(y[:, np.newaxis])
    return x, y, z

@image_comparison(baseline_images=['contour_hatching'])
def test_contour_hatching():
    x, y, z = contour_dat()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(x, y, z, hatches=['-', '/', '\\', '//'],
                      cmap=plt.get_cmap('gray'),
                      extend='both', alpha=0.5)

@image_comparison(baseline_images=['contour_colorbar'])
def test_contour_colorbar():
    x, y, z = contour_dat()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(x, y, z, levels=np.arange(-1.8, 1.801, 0.2),
                      cmap=plt.get_cmap('RdBu'),
                      vmin=-0.6,
                      vmax=0.6,
                      extend='both')
    cs1 = ax.contour(x, y, z, levels=np.arange(-2.2, -0.599, 0.2),
                              colors=['y'],
                              linestyles='solid',
                              linewidths=2)
    cs2 = ax.contour(x, y, z, levels=np.arange(0.6, 2.2, 0.2),
                              colors=['c'],
                              linewidths=2)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.add_lines(cs1)
    cbar.add_lines(cs2, erase=False)


@image_comparison(baseline_images=['hist2d'])
def test_hist2d():
    np.random.seed(0)
    #make it not symetric in case we switch x and y axis
    x=np.random.randn(100)*2+5
    y = np.random.randn(100)-2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(x,y,bins=10)


@image_comparison(baseline_images=['hist2d_transpose'])
def test_hist2d_transpose():
    np.random.seed(0)
    #make sure the the output from np.histogram is transposed before
    #passing to pcolorfast
    x=np.array([5]*100)
    y = np.random.randn(100)-2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(x,y,bins=10)


@image_comparison(baseline_images=['scatter'])
def test_scatter_plot():
    ax = plt.axes()
    ax.scatter([3, 4, 2, 6], [2, 5, 2, 3], c=['r', 'y', 'b', 'lime'], s=[24, 15, 19, 29])

@cleanup
def test_as_mpl_axes_api():
    # tests the _as_mpl_axes api
    from matplotlib.projections.polar import PolarAxes
    import matplotlib.axes as maxes

    class Polar(object):
        def __init__(self):
            self.theta_offset = 0

        def _as_mpl_axes(self):
            # implement the matplotlib axes interface
            return PolarAxes, {'theta_offset': self.theta_offset}
    prj = Polar()
    prj2 = Polar()
    prj2.theta_offset = np.pi
    prj3 = Polar()

    # testing axes creation with plt.axes
    ax = plt.axes([0, 0, 1, 1], projection=prj)
    assert type(ax) == PolarAxes, \
           'Expected a PolarAxes, got %s' % type(ax)
    ax_via_gca = plt.gca(projection=prj)
    assert ax_via_gca is ax
    plt.close()

    # testing axes creation with gca
    ax = plt.gca(projection=prj)
    assert type(ax) == maxes._subplot_classes[PolarAxes], \
           'Expected a PolarAxesSubplot, got %s' % type(ax)
    ax_via_gca = plt.gca(projection=prj)
    assert ax_via_gca is ax
    # try getting the axes given a different polar projection
    ax_via_gca = plt.gca(projection=prj2)
    assert ax_via_gca is not ax
    assert ax.get_theta_offset() == 0, ax.get_theta_offset()
    assert ax_via_gca.get_theta_offset() == np.pi, ax_via_gca.get_theta_offset()
    # try getting the axes given an == (not is) polar projection
    ax_via_gca = plt.gca(projection=prj3)
    assert ax_via_gca is ax
    plt.close()

    # testing axes creation with subplot
    ax = plt.subplot(121, projection=prj)
    assert type(ax) == maxes._subplot_classes[PolarAxes], \
           'Expected a PolarAxesSubplot, got %s' % type(ax)
    plt.close()

@image_comparison(baseline_images=['log_scales'])
def test_log_scales():
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(np.log(np.linspace(0.1, 100)))
    ax.set_yscale('log', basey=5.5)
    ax.invert_yaxis()
    ax.set_xscale('log', basex=9.0)


@image_comparison(baseline_images=['stackplot_test_image'])
def test_stackplot():
    fig = plt.figure()
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    ax = fig.add_subplot(1, 1, 1)
    ax.stackplot(x, y1, y2, y3)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))


@image_comparison(baseline_images=['stackplot_test_baseline'],
                  remove_text=True)
def test_stackplot_baseline():
    np.random.seed(0)
    def layers(n, m):
        def bump(a):
            x = 1 / (.1 + np.random.random())
            y = 2 * np.random.random() - .5
            z = 10 / (.1 + np.random.random())
            for i in range(m):
                w = (i / float(m) - y) * z
                a[i] += x * np.exp(-w * w)
        a = np.zeros((m, n))
        for i in range(n):
            for j in range(5):
                bump(a[:, i])
        return a

    d=layers(3, 100)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.stackplot(range(100), d.T, baseline='zero')

    plt.subplot(2, 2, 2)
    plt.stackplot(range(100), d.T, baseline='sym')

    plt.subplot(2, 2, 3)
    plt.stackplot(range(100), d.T, baseline='wiggle')

    plt.subplot(2, 2, 4)
    plt.stackplot(range(100), d.T, baseline='weighted_wiggle')


@image_comparison(baseline_images=['boxplot'])
def test_boxplot():
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # show 1 boxplot with mpl medians/conf. interfals, 1 with manual values
    ax.boxplot([x, x], bootstrap=10000, usermedians=[None, 1.0],
               conf_intervals=[None, (-1.0, 3.5)], notch=1)
    ax.set_ylim((-30, 30))

@image_comparison(baseline_images=['boxplot_no_inverted_whisker'],
                  remove_text=True, extensions=['png'],
                  savefig_kwarg={'dpi': 40})
def test_boxplot_no_weird_whisker():
    x = np.array([3, 9000, 150, 88, 350, 200000, 1400, 960],
                dtype=np.float64)
    ax1 = plt.axes()
    ax1.boxplot(x)
    ax1.set_yscale('log')
    ax1.yaxis.grid(False, which='minor')
    ax1.xaxis.grid(False)

@image_comparison(baseline_images=['errorbar_basic',
                                   'errorbar_mixed'])
def test_errorbar():
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)

    yerr = 0.1 + 0.2*np.sqrt(x)
    xerr = 0.1 + yerr

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title('Vert. symmetric')

    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)

    ax = axs[0,1]
    ax.errorbar(x, y, xerr=xerr, fmt='o', alpha=0.4)
    ax.set_title('Hor. symmetric w/ alpha')

    ax = axs[1,0]
    ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
    ax.set_title('H, V asymmetric')

    ax = axs[1,1]
    ax.set_yscale('log')
    # Here we have to be careful to keep all y values positive:
    ylower = np.maximum(1e-2, y - yerr)
    yerr_lower = y - ylower

    ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
                        fmt='o', ecolor='g', capthick=2)
    ax.set_title('Mixed sym., log y')

    fig.suptitle('Variable errorbars')

@image_comparison(baseline_images=['hist_stacked_stepfilled'])
def test_hist_stacked_stepfilled():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( (d1, d2), histtype="stepfilled", stacked=True)


@image_comparison(baseline_images=['hist_offset'])
def test_hist_offset():
    # make some data
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(d1, bottom=5)
    ax.hist(d2, bottom=15)


@image_comparison(baseline_images=['hist_step'], extensions=['png'], remove_text=True)
def test_hist_step():
    # make some data
    d1 = np.linspace(1, 3, 20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( d1, histtype="step")
    ax.set_ylim(0, 10)
    ax.set_xlim(-1, 5)


@image_comparison(baseline_images=['hist_stacked_weights'])
def test_hist_stacked_weighted():
    # make some data
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    w1 = np.linspace(0.01, 3.5, 50)
    w2 = np.linspace(0.05, 2., 20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( (d1, d2), weights=(w1,w2), histtype="stepfilled", stacked=True)

@cleanup
def test_stem_args():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = range(10)
    y = range(10)

    # Test the call signatures
    ax.stem(y)
    ax.stem(x, y)
    ax.stem(x, y, 'r--')
    ax.stem(x, y, 'r--', basefmt='b--')

@image_comparison(baseline_images=['hist_stacked_stepfilled_alpha'])
def test_hist_stacked_stepfilled_alpha():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( (d1, d2), histtype="stepfilled", stacked=True, alpha=0.5)

@image_comparison(baseline_images=['hist_stacked_step'])
def test_hist_stacked_step():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( (d1, d2), histtype="step", stacked=True)


@image_comparison(baseline_images=['hist_stacked_normed'])
def test_hist_stacked_normed():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist((d1, d2), stacked=True, normed=True)


@image_comparison(baseline_images=['hist_stacked_bar'])
def test_hist_stacked_bar():
    # make some data
    d = [[100, 100, 100, 100, 200, 320, 450, 80, 20, 600, 310, 800], [20, 23, 50, 11, 100, 420], [120, 120, 120, 140, 140, 150, 180], [60, 60, 60, 60, 300, 300, 5, 5, 5, 5, 10, 300], [555, 555, 555, 30, 30, 30, 30, 30, 100, 100, 100, 100, 30, 30], [30, 30, 30, 30, 400, 400, 400, 400, 400, 400, 400, 400]]
    colors = [(0.5759849696758961, 1.0, 0.0), (0.0, 1.0, 0.350624650815206), (0.0, 1.0, 0.6549834156005998), (0.0, 0.6569064625276622, 1.0), (0.28302699607823545, 0.0, 1.0), (0.6849123462299822, 0.0, 1.0)]
    labels = ['green', 'orange', ' yellow', 'magenta', 'black']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(d, bins=10, histtype='barstacked', align='mid', color=colors, label=labels)
    ax.legend(loc='upper right', bbox_to_anchor = (1.0, 1.0), ncol=1)

@image_comparison(baseline_images=['transparent_markers'], remove_text=True)
def test_transparent_markers():
    np.random.seed(0)
    data = np.random.random(50)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, 'D', mfc='none', markersize=100)

@image_comparison(baseline_images=['mollweide_grid'], remove_text=True)
def test_mollweide_grid():
    # test that both horizontal and vertical gridlines appear on the Mollweide
    # projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.grid()

@cleanup
def test_mollweide_forward_inverse_closure():
    # test that the round-trip Mollweide forward->inverse transformation is an
    # approximate identity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')

    # set up 1-degree grid in longitude, latitude
    lon = np.linspace(-np.pi, np.pi, 360)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 180)
    lon, lat = np.meshgrid(lon, lat)
    ll = np.vstack((lon.flatten(), lat.flatten())).T

    # perform forward transform
    xy = ax.transProjection.transform(ll)

    # perform inverse transform
    ll2 = ax.transProjection.inverted().transform(xy)

    # compare
    np.testing.assert_array_almost_equal(ll, ll2, 3)

@cleanup
def test_mollweide_inverse_forward_closure():
    # test that the round-trip Mollweide inverse->forward transformation is an
    # approximate identity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')

    # set up grid in x, y
    x = np.linspace(0, 1, 500)
    x, y = np.meshgrid(x, x)
    xy = np.vstack((x.flatten(), y.flatten())).T

    # perform inverse transform
    ll = ax.transProjection.inverted().transform(xy)

    # perform forward transform
    xy2 = ax.transProjection.transform(ll)

    # compare
    np.testing.assert_array_almost_equal(xy, xy2, 3)


@image_comparison(baseline_images=['test_alpha'], remove_text=True)
def test_alpha():
    np.random.seed(0)
    data = np.random.random(50)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # alpha=.5 markers, solid line
    ax.plot(data, '-D', color=[1, 0, 0], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # everything solid by kwarg
    ax.plot(data + 2, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10,
            alpha=1)

    # everything alpha=.5 by kwarg
    ax.plot(data + 4, '-D', color=[1, 0, 0], mfc=[1, 0, 0],
            markersize=20, lw=10,
            alpha=.5)

    # everything alpha=.5 by colors
    ax.plot(data + 6, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # alpha=.5 line, solid markers
    ax.plot(data + 8, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0],
            markersize=20, lw=10)


@image_comparison(baseline_images=['eventplot'], remove_text=True)
def test_eventplot():
    '''
    test that eventplot produces the correct output
    '''
    np.random.seed(0)

    data1 = np.random.random([32, 20]).tolist()
    data2 = np.random.random([6, 20]).tolist()
    data = data1 + data2
    num_datasets = len(data)

    colors1 = [[0, 1, .7]] * len(data1)
    colors2 = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, .75, 0],
               [1, 0, 1],
               [0, 1, 1]]
    colors = colors1 + colors2

    lineoffsets1 = 12 + np.arange(0, len(data1)) * .33
    lineoffsets2 = [-15, -3, 1, 1.5, 6, 10]
    lineoffsets = lineoffsets1.tolist() + lineoffsets2

    linelengths1 = [.33] * len(data1)
    linelengths2 = [5, 2, 1, 1, 3, 1.5]
    linelengths = linelengths1 + linelengths2

    fig = plt.figure()
    axobj = fig.add_subplot(111)
    colls = axobj.eventplot(data, colors=colors, lineoffsets=lineoffsets,
                            linelengths=linelengths)

    num_collections = len(colls)
    np.testing.assert_equal(num_collections, num_datasets)

@image_comparison(baseline_images=['vertex_markers'], extensions=['png'],
                  remove_text=True)
def test_vertex_markers():
    data = range(10)
    marker_as_tuple = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    marker_as_list = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, linestyle='', marker=marker_as_tuple, mfc='k')
    ax.plot(data[::-1], linestyle='', marker=marker_as_list, mfc='b')
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])

@image_comparison(baseline_images=['vline_hline_zorder',
                                   'errorbar_zorder'])
def test_eb_line_zorder():
    x = range(10)

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, lw=10, zorder=5)
    ax.axhline(1, color='red', lw=10, zorder=1)
    ax.axhline(5, color='green', lw=10, zorder=10)
    ax.axvline(7, color='m', lw=10, zorder=7)
    ax.axvline(2, color='k', lw=10, zorder=3)

    ax.set_title("axvline and axhline zorder test")

    # Now switch to a more OO interface to exercise more features.
    fig = plt.figure()
    ax = fig.gca()
    x = range(10)
    y = np.zeros(10)
    yerr = range(10)
    ax.errorbar(x, y, yerr=yerr, zorder=5, lw=5, color='r')
    for j in range(10):
        ax.axhline(j, lw=5, color='k', zorder=j)
        ax.axhline(-j, lw=5, color='k', zorder=j)

    ax.set_title("errorbar zorder test")


@image_comparison(baseline_images=['step_linestyle'], remove_text=True)
def test_step_linestyle():
    x = y = np.arange(10)

    # First illustrate basic pyplot interface, using defaults where possible.
    fig, ax_lst = plt.subplots(2, 2)
    ax_lst = ax_lst.flatten()

    ln_styles = ['-', '--', '-.', ':']

    for ax, ls in zip(ax_lst, ln_styles):
        ax.step(x, y, lw=5, linestyle=ls, where='pre')
        ax.step(x, y + 1, lw=5, linestyle=ls, where='mid')
        ax.step(x, y + 2, lw=5, linestyle=ls, where='post')
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 7])


@image_comparison(baseline_images=['mixed_collection'], remove_text=True)
def test_mixed_collection():
    from matplotlib import patches
    from matplotlib import collections

    x = range(10)

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    c = patches.Circle((8, 8), radius=4, facecolor='none', edgecolor='green')

    # PDF can optimize this one
    p1 = collections.PatchCollection([c], match_original=True)
    p1.set_offsets([[0, 0], [24, 24]])
    p1.set_linewidths([1, 5])

    # PDF can't optimize this one, because the alpha of the edge changes
    p2 = collections.PatchCollection([c], match_original=True)
    p2.set_offsets([[48, 0], [-32, -16]])
    p2.set_linewidths([1, 5])
    p2.set_edgecolors([[0, 0, 0.1, 1.0], [0, 0, 0.1, 0.5]])

    ax.patch.set_color('0.5')
    ax.add_collection(p1)
    ax.add_collection(p2)

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)

@cleanup
def test_subplot_key_hash():
    ax = plt.subplot(np.float64(5.5), np.int64(1), np.float64(1.2))
    ax.twinx()
    assert_equal((5, 1, 0, None), ax.get_subplotspec().get_geometry())


@image_comparison(baseline_images=['specgram_freqs'], remove_text=True,
                  extensions=['png'])
def test_specgram_freqs():
    n = 10000
    Fs = 100.

    fstims1 = [Fs/4, Fs/5, Fs/11]
    fstims2 = [Fs/4.7, Fs/5.6, Fs/11.9]

    NFFT = int(1000 * Fs / min(fstims1 + fstims2))
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    x = np.arange(0, n, 1/Fs)

    y1 = np.zeros(x.size)
    y2 = np.zeros(x.size)
    for fstim1, fstim2 in zip(fstims1, fstims2):
        y1 += np.sin(fstim1 * x * np.pi * 2)
        y2 += np.sin(fstim2 * x * np.pi * 2)
    y = np.hstack([y1, y2])

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    spec1 = ax1.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='default')
    spec2 = ax2.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='onesided')
    spec3 = ax3.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='twosided')


@image_comparison(baseline_images=['specgram_noise'], remove_text=True,
                  extensions=['png'])
def test_specgram_noise():
    np.random.seed(0)

    n = 10000
    Fs = 100.

    NFFT = int(1000 * Fs / 11)
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    y1 = np.random.standard_normal(n)
    y2 = np.random.rand(n)
    y = np.hstack([y1, y2])

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    spec1 = ax1.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='default')
    spec2 = ax2.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='onesided')
    spec3 = ax3.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                         sides='twosided')


@image_comparison(baseline_images=['psd_freqs'], remove_text=True,
                  extensions=['png'])
def test_psd_freqs():
    n = 10000
    Fs = 100.

    fstims1 = [Fs/4, Fs/5, Fs/11]
    fstims2 = [Fs/4.7, Fs/5.6, Fs/11.9]

    NFFT = int(1000 * Fs / min(fstims1 + fstims2))
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    x = np.arange(0, n, 1/Fs)

    y1 = np.zeros(x.size)
    y2 = np.zeros(x.size)
    for fstim1, fstim2 in zip(fstims1, fstims2):
        y1 += np.sin(fstim1 * x * np.pi * 2)
        y2 += np.sin(fstim2 * x * np.pi * 2)
    y = np.hstack([y1, y2])

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    psd1 = ax1.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='default')
    psd2 = ax2.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='onesided')
    psd3 = ax3.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='twosided')

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')


@image_comparison(baseline_images=['psd_noise'], remove_text=True,
                  extensions=['png'])
def test_psd_noise():
    np.random.seed(0)

    n = 10000
    Fs = 100.

    NFFT = int(1000 * Fs / 11)
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    y1 = np.random.standard_normal(n)
    y2 = np.random.rand(n)
    y = np.hstack([y1, y2])

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    psd1 = ax1.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='default')
    psd2 = ax2.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='onesided')
    psd3 = ax3.psd(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='twosided')

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')


@image_comparison(baseline_images=['csd_freqs'], remove_text=True,
                  extensions=['png'])
def test_csd_freqs():
    n = 10000
    Fs = 100.

    fstims1 = [Fs/4, Fs/5, Fs/11]
    fstims2 = [Fs/4.7, Fs/5.6, Fs/11.9]

    NFFT = int(1000 * Fs / min(fstims1 + fstims2))
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    x = np.arange(0, n, 1/Fs)

    y1 = np.zeros(x.size)
    y2 = np.zeros(x.size)
    for fstim1, fstim2 in zip(fstims1, fstims2):
        y1 += np.sin(fstim1 * x * np.pi * 2)
        y2 += np.sin(fstim2 * x * np.pi * 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    csd1 = ax1.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='default')
    csd2 = ax2.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='onesided')
    csd3 = ax3.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='twosided')

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')


@image_comparison(baseline_images=['csd_noise'], remove_text=True,
                  extensions=['png'])
def test_csd_noise():
    np.random.seed(0)

    n = 10000
    Fs = 100.

    NFFT = int(1000 * Fs / 11)
    noverlap = int(NFFT / 2)
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))

    y1 = np.random.standard_normal(n)
    y2 = np.random.rand(n)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    csd1 = ax1.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='default')
    csd2 = ax2.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='onesided')
    csd3 = ax3.csd(y1, y2, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to,
                   sides='twosided')

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')


@image_comparison(baseline_images=['twin_spines'], remove_text=True,
                  extensions=['png'])
def test_twin_spines():

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.itervalues():
            sp.set_visible(False)

    fig = plt.figure(figsize=(4, 3))
    fig.subplots_adjust(right=0.75)

    host = fig.add_subplot(111)
    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p1, = host.plot([0, 1, 2], [0, 1, 2], "b-")
    p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-")
    p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-")

    host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)


@cleanup
def test_vline_limit():
    fig = plt.figure()
    ax = fig.gca()
    ax.axvline(0.5)
    ax.plot([-0.1, 0, 0.2, 0.1])
    (ymin, ymax) = ax.get_ylim()
    assert ymin == -0.1
    assert ymax == 0.25


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
