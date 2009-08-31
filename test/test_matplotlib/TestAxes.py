#=======================================================================
"""The Axes unit-test class implementation."""
#=======================================================================

from mplTest import MplTestCase, units
from matplotlib.testing.decorators import knownfailureif

#=======================================================================
# Add import modules below.
import matplotlib
matplotlib.use( "Agg", warn = False )

import pylab
import numpy as npy
from datetime import datetime
#
#=======================================================================

#=======================================================================
class TestAxes( MplTestCase ):
   """Test the various axes non-plotting methods."""

   # Uncomment any appropriate tags
   tags = [
            # 'gui',        # requires the creation of a gui window
            'agg',        # uses agg in the backend
            'agg-only',   # uses only agg in the backend
            # 'wx',         # uses wx in the backend
            # 'qt',         # uses qt in the backend
            # 'ps',         # uses the postscript backend
            # 'units',      # uses units in the test
            'PIL',        # uses PIL for image comparison
          ]

   #--------------------------------------------------------------------
   def setUp( self ):
      """Setup any data needed for the unit test."""
      units.register()

   #--------------------------------------------------------------------
   def tearDown( self ):
      """Clean-up any generated files here."""
      pass

   #--------------------------------------------------------------------
   def test_empty_datetime( self ):
      """Test plotting empty axes with dates along one axis."""
      if 1:
         raise RuntimeError('error forced to test buildbot error reporting')
      fname = self.outFile( "empty_datetime.png" )

      t0 = datetime(2009, 1, 20)
      tf = datetime(2009, 1, 21)

      fig = pylab.figure()
      pylab.axvspan( t0, tf, facecolor="blue", alpha=0.25 )
      fig.autofmt_xdate()

      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------
   def test_formatter_ticker( self ):
      """Test Some formatter and ticker issues."""

      # This essentially test to see if user specified labels get overwritten
      # by the auto labeler functionality of the axes.
      xdata = [ x*units.sec for x in range(10) ]
      ydata1 = [ (1.5*y - 0.5)*units.km for y in range(10) ]
      ydata2 = [ (1.75*y - 1.0)*units.km for y in range(10) ]

      fname = self.outFile( "formatter_ticker_001.png" )
      fig = pylab.figure()
      ax = pylab.subplot( 111 )
      ax.set_xlabel( "x-label 001" )
      fig.savefig( fname )
      self.checkImage( fname )

      fname = self.outFile( "formatter_ticker_002.png" )
      ax.plot( xdata, ydata1, color='blue', xunits="sec" )
      fig.savefig( fname )
      self.checkImage( fname )

      fname = self.outFile( "formatter_ticker_003.png" )
      ax.set_xlabel( "x-label 003" )
      fig.savefig( fname )
      self.checkImage( fname )

      fname = self.outFile( "formatter_ticker_004.png" )
      ax.plot( xdata, ydata2, color='green', xunits="hour" )
      ax.set_xlabel( "x-label 004" )
      fig.savefig( fname )
      self.checkImage( fname )

      # See SF bug 2846058
      # https://sourceforge.net/tracker/?func=detail&aid=2846058&group_id=80706&atid=560720
      fname = self.outFile( "formatter_ticker_005.png" )
      ax.set_xlabel( "x-label 005" )
      ax.autoscale_view()
      fig.savefig( fname )
      self.checkImage( fname )


