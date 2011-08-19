import numpy as np
from numpy import ma
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif
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

@image_comparison(baseline_images=['offset_points'])
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
@image_comparison(baseline_images=['polar_coords'])
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

@image_comparison(baseline_images=['fill_units'])
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
    from nose.tools import assert_raises
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

    #NOTE: resolution=1 really should be the default
    plt.subplot( 111, polar=True, resolution=1 )
    plt.polar( [179*D2R, -179*D2R], [0.2, 0.1], "b.-" )
    plt.polar( [179*D2R,  181*D2R], [0.2, 0.1], "g.-" )
    plt.rgrids( [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] )

    fig = plt.figure()

    #NOTE: resolution=1 really should be the default
    plt.subplot( 111, polar=True, resolution=1 )
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


@image_comparison(baseline_images=['hexbin_extent'])
def test_hexbin_extent():
    # this test exposes sf bug 2856228
    fig = plt.figure()

    ax = fig.add_subplot(111)
    data = np.arange(2000.)/2000.
    data.shape = 2, 1000
    x, y = data

    ax.hexbin(x, y, extent=[.1, .3, .6, .7])

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

@image_comparison(baseline_images=['imshow'])
def test_imshow():
    #Create a NxN image
    N=100
    (x,y) = np.indices((N,N))
    x -= N/2
    y -= N/2
    r = np.sqrt(x**2+y**2-x*y)

    #Create a contour plot at N/4 and extract both the clip path and transform
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(r)

@image_comparison(baseline_images=['imshow_clip'], tol=1e-2)
def test_imshow_clip():
    # As originally reported by Gellule Xg <gellule.xg@free.fr>

    #Create a NxN image
    N=100
    (x,y) = np.indices((N,N))
    x -= N/2
    y -= N/2
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

@image_comparison(baseline_images=['polycollection_joinstyle'])
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
    ax.set_xticks([])
    ax.set_yticks([])

@image_comparison(baseline_images=['fill_between_interpolate'], tol=1e-2)
def test_fill_between_interpolate():
    x = np.arange(0.0, 2, 0.02)
    y1 = np.sin(2*np.pi*x)
    y2 = 1.2*np.sin(4*np.pi*x)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(x, y1, x, y2, color='black')
    ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='green', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2<=y1, facecolor='red', interpolate=True)

    # Test support for masked arrays.
    y2 = np.ma.masked_greater(y2, 1.0)
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

@image_comparison(baseline_images=['symlog2'])
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
    
@image_comparison(baseline_images=['pcolormesh'], tol=0.02)
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
    ax.set_title('lw=0.5')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(132)
    ax.pcolormesh(Qx,Qz,Z, lw=3, edgecolors='k')
    ax.set_title('lw=3')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(133)
    ax.pcolormesh(Qx,Qz,Z, shading="gouraud")
    ax.set_title('gouraud')
    ax.set_xticks([])
    ax.set_yticks([])
    

@image_comparison(baseline_images=['canonical'])
def test_canonical():
    fig, ax = plt.subplots()
    ax.plot([1,2,3])


@image_comparison(baseline_images=['arc_ellipse'])
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

@image_comparison(baseline_images=['markevery'])
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

@image_comparison(baseline_images=['markevery_line'])
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

    
if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
