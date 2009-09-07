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
