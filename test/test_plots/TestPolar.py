#=======================================================================
"""The Polar unit-test class implementation."""
#=======================================================================

from mplTest import *

#=======================================================================
# Add import modules below.
import matplotlib
matplotlib.use( "Agg", warn = False )

import pylab
import numpy as npy
#
#=======================================================================

#=======================================================================
class TestPolar( MplTestCase ):
   """Polar unit test class."""

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
   def test_polar_units( self ):
      """Test polar plots with unitized data."""

      fname = self.outFile( "polar_units.png" )

      pi = npy.pi
      deg = units.UnitDbl( 1.0, "deg" )

      x1 = [ pi/6.0, pi/4.0, pi/3.0, pi/2.0 ]
      x2 = [ 30.0*deg, 45.0*deg, 60.0*deg, 90.0*deg ]

      y1 = [ 1.0, 2.0, 3.0, 4.0]
      y2 = [ 4.0, 3.0, 2.0, 1.0 ]

      fig = pylab.figure()

      pylab.polar( x2, y1, color = "blue" )

      # polar( x2, y1, color = "red", xunits="rad" )
      # polar( x2, y2, color = "green" )

      fig.savefig( fname )
      self.checkImage( fname )

