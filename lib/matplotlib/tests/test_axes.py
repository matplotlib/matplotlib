import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
import pylab

@knownfailureif('indeterminate', "Fails due to SF bug 2850075")
@image_comparison(baseline_images=['empty_datetime'])
def test_empty_datetime():
    """Test plotting empty axes with dates along one axis."""
    from datetime import datetime

    t0 = datetime(2009, 1, 20)
    tf = datetime(2009, 1, 21)

    fig = pylab.figure()
    pylab.axvspan( t0, tf, facecolor="blue", alpha=0.25 )
    fig.autofmt_xdate()

    fig.savefig( 'empty_datetime' )

@image_comparison(baseline_images=['formatter_ticker_001',
                                   'formatter_ticker_002',
                                   'formatter_ticker_003',
                                   'formatter_ticker_004',
                                   'formatter_ticker_005',
                                   ])
def test_formatter_ticker():
    """Test Some formatter and ticker issues."""
    import matplotlib.testing.jpl_units as units
    def register_units():
        """Register the unit conversion classes with matplotlib."""
        import matplotlib.units as munits
        import matplotlib.testing.jpl_units as jpl_units
        from matplotlib.testing.jpl_units.Duration import Duration
        from matplotlib.testing.jpl_units.Epoch import Epoch
        from matplotlib.testing.jpl_units.UnitDbl import UnitDbl

        from matplotlib.testing.jpl_units.StrConverter import StrConverter
        from matplotlib.testing.jpl_units.EpochConverter import EpochConverter
        from matplotlib.testing.jpl_units.UnitDblConverter import UnitDblConverter

        munits.registry[ str ] = StrConverter()
        munits.registry[ Epoch ] = EpochConverter()
        munits.registry[ UnitDbl ] = UnitDblConverter()
    register_units()

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
