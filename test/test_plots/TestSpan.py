#=======================================================================
"""The Span unit-test class implementation."""
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
class TestSpan( MplTestCase ):
   """Test the various axes spanning methods."""

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
   def test_axvspan_epoch( self ):
      """Test the axvspan method with Epochs."""

      fname = self.outFile( "axvspan_epoch.png" )

      # generate some data
      t0 = units.Epoch( "ET", dt=datetime(2009, 1, 20) )
      tf = units.Epoch( "ET", dt=datetime(2009, 1, 21) )

      dt = units.Duration( "ET", units.day.convert( "sec" ) )

      fig = pylab.figure()

      pylab.axvspan( t0, tf, facecolor="blue", alpha=0.25 )

      ax = pylab.gca()
      ax.set_xlim( t0 - 5.0*dt, tf + 5.0*dt )

      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------
   def test_axhspan_epoch( self ):
      """Test the axhspan method with Epochs."""

      fname = self.outFile( "axhspan_epoch.png" )

      # generate some data
      t0 = units.Epoch( "ET", dt=datetime(2009, 1, 20) )
      tf = units.Epoch( "ET", dt=datetime(2009, 1, 21) )

      dt = units.Duration( "ET", units.day.convert( "sec" ) )

      fig = pylab.figure()

      pylab.axhspan( t0, tf, facecolor="blue", alpha=0.25 )

      ax = pylab.gca()
      ax.set_ylim( t0 - 5.0*dt, tf + 5.0*dt )

      fig.savefig( fname )
      self.checkImage( fname )
