#=======================================================================
"""The Fill unit-test class implementation."""
#=======================================================================

from mplTest import *

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
class TestFill( MplTestCase ):
   """Test the various axes fill methods."""

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
   def test_fill_units( self ):
      """Test the fill method with unitized-data."""

      fname = self.outFile( "fill_units.png" )

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
      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------

