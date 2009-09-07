#=======================================================================
"""The Tickers unit-test class implementation."""
#=======================================================================

from mplTest import *

#=======================================================================
# Add import modules below.
import matplotlib
matplotlib.use( "Agg", warn = False )

import pylab
import numpy as npy
from datetime import datetime

import dateutil
import matplotlib.dates as mpldates
import matplotlib.ticker as ticker
#
#=======================================================================

#=======================================================================
class TestTickers( MplTestCase ):
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
   def test_DateFormatter( self ):
      """Test DateFormatter"""

      # Lets make sure that DateFormatter will allow us to have tick marks
      # at intervals of fractional seconds.
      fname = self.outFile( "DateFormatter_fractionalSeconds.png" )

      t0 = datetime( 2001, 1, 1, 0, 0, 0 )
      tf = datetime( 2001, 1, 1, 0, 0, 1 )

      fig = pylab.figure()
      ax = pylab.subplot( 111 )
      ax.set_autoscale_on( True )
      ax.plot( [t0, tf], [0.0, 1.0], marker='o' )

      # rrule = mpldates.rrulewrapper( dateutil.rrule.YEARLY, interval=500 )
      # locator = mpldates.RRuleLocator( rrule )
      # ax.xaxis.set_major_locator( locator )
      # ax.xaxis.set_major_formatter( mpldates.AutoDateFormatter(locator) )

      ax.autoscale_view()
      fig.autofmt_xdate()

      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------
