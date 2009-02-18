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
   def test_polar_wrap( self ):
      """Test polar plots where data crosses 0 degrees."""

      fname = self.outFile( "polar_wrap_180.png" )

      D2R = npy.pi / 180.0

      fig = pylab.figure()

      #NOTE: resolution=1 really should be the default
      pylab.subplot( 111, polar=True, resolution=1 )
      pylab.polar( [179*D2R, -179*D2R], [0.2, 0.1], "b.-" )
      pylab.polar( [179*D2R,  181*D2R], [0.2, 0.1], "g.-" )
      pylab.rgrids( [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] )

      fig.savefig( fname )
      self.checkImage( fname )


      fname = self.outFile( "polar_wrap_360.png" )

      fig = pylab.figure()

      #NOTE: resolution=1 really should be the default
      pylab.subplot( 111, polar=True, resolution=1 )
      pylab.polar( [2*D2R, -2*D2R], [0.2, 0.1], "b.-" )
      pylab.polar( [2*D2R,  358*D2R], [0.2, 0.1], "g.-" )
      pylab.polar( [358*D2R,  2*D2R], [0.2, 0.1], "r.-" )
      pylab.rgrids( [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] )

      fig.savefig( fname )
      self.checkImage( fname )

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

