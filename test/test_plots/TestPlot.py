#=======================================================================
"""The Plot unit-test class implementation."""
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
class TestPlot( MplTestCase ):
   """Plot unit test class."""

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
      pass

   #--------------------------------------------------------------------
   def tearDown( self ):
      """Clean-up any generated files here."""
      pass

   #--------------------------------------------------------------------
   def test_const_xy( self ):
      """Test constant xy data."""

      fname = self.outFile( "const_xy.png" )
      fig = pylab.figure()

      pylab.subplot( 311 )
      pylab.plot( npy.arange(10), npy.ones( (10,) ) )

      pylab.subplot( 312 )
      pylab.plot( npy.ones( (10,) ), npy.arange(10) )

      pylab.subplot( 313 )
      pylab.plot( npy.ones( (10,) ), npy.ones( (10,) ), 'o' )

      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------

