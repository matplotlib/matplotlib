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
   def testBasicAnnotate( self ):
      """Basic Annotations"""

      # Setup some data
      t = npy.arange( 0.0, 5.0, 0.01 )
      s = npy.cos( 2.0*npy.pi * t )

      # Offset Points
      fname = self.outFile( "offset_points.png" )

      fig = figure()
      ax = fig.add_subplot( 111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5) )
      line, = ax.plot( t, s, lw=3, color='purple' )

      ax.annotate( 'local max', xy=(3, 1), xycoords='data',
                  xytext=(3, 3), textcoords='offset points' )

      fig.savefig( fname )
      self.checkImage( fname )

   #--------------------------------------------------------------------
   def testPolarAnnotations( self ):
      """Polar Plot Annotations"""

      # you can specify the xypoint and the xytext in different
      # positions and coordinate systems, and optionally turn on a
      # connecting line and mark the point with a marker.  Annotations
      # work on polar axes too.  In the example below, the xy point is
      # in native coordinates (xycoords defaults to 'data').  For a
      # polar axes, this is in (theta, radius) space.  The text in this
      # example is placed in the fractional figure coordinate system.
      # Text keyword args like horizontal and vertical alignment are
      # respected

      # Setup some data
      r = npy.arange(0.0, 1.0, 0.001 )
      theta = 2.0 * 2.0 * npy.pi * r

      fname = self.outFile( "polar_axes.png" )

      fig = figure()
      ax = fig.add_subplot( 111, polar=True )
      line, = ax.plot( theta, r, color='#ee8d18', lw=3 )

      ind = 800
      thisr, thistheta = r[ind], theta[ind]
      ax.plot([thistheta], [thisr], 'o')
      ax.annotate('a polar annotation',
                  xy=(thistheta, thisr),  # theta, radius
                  xytext=(0.05, 0.05),    # fraction, fraction
                  textcoords='figure fraction',
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  )

      fig.savefig( fname )
      self.checkImage( fname )

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

