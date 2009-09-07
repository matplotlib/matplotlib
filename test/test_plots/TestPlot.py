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
   def test_shaped_data( self ):
      """Test numpy shaped data."""

      xdata = npy.array([[ 0.53295185,  0.23052951,  0.19057629,  0.66724975,  0.96577916,
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

      fname = self.outFile( "shaped_data.png" )

      y1 = npy.arange( 10 )
      y1.shape = 1, 10

      y2 = npy.arange( 10 )
      y2.shape = 10, 1

      fig = pylab.figure()
      pylab.subplot( 411 )
      pylab.plot( y1 )
      pylab.subplot( 412 )
      pylab.plot( y2 )

      pylab.subplot( 413 )
      try:
         pylab.plot( y1, y2 )
      except:
         # This should fail
         pass
      else:
         self.fail( "Failed to raise an exception for mis-matched dimensions." )

      pylab.subplot( 414 )
      pylab.plot( xdata[:,1], xdata[1,:], 'o' )

      fig.savefig( fname )
      self.checkImage( fname )

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

