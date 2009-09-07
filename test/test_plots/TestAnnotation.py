#=======================================================================
"""The Annotation unite-test class implementation."""
#=======================================================================

from mplTest import *

#=======================================================================
# Add import modules below.
import matplotlib
matplotlib.use( "Agg", warn = False )

from matplotlib.pyplot import figure
from matplotlib.patches import Ellipse
import numpy as npy
#
#=======================================================================

#=======================================================================
class TestAnnotation( MplTestCase ):
   """Annotation unit test class."""

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
   def testPolarCoordAnnotations( self ):
      """Polar Coordinate Annotations"""

      # You can also use polar notation on a catesian axes.  Here the
      # native coordinate system ('data') is cartesian, so you need to
      # specify the xycoords and textcoords as 'polar' if you want to
      # use (theta, radius)

      el = Ellipse((0,0), 10, 20, facecolor='r', alpha=0.5)

      fname = self.outFile( "polar_coords.png" )

      fig = figure()
      ax = fig.add_subplot( 111, aspect='equal' )

      ax.add_artist( el )
      el.set_clip_box( ax.bbox )

      ax.annotate('the top',
                   xy=(npy.pi/2., 10.),      # theta, radius
                   xytext=(npy.pi/3, 20.),   # theta, radius
                   xycoords='polar',
                   textcoords='polar',
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   clip_on=True, # clip to the axes bounding box
      )

      ax.set_xlim( -20, 20 )
      ax.set_ylim( -20, 20 )
      fig.savefig( fname )
      self.checkImage( fname )

