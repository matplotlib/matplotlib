#=======================================================================
"""The Cookbook unit-test class implementation."""
#=======================================================================

from mplTest import *

#=======================================================================
# Add import modules below.
import matplotlib
matplotlib.use( "Agg", warn = False )

import numpy as npy
import matplotlib.cbook as cbook
#
#=======================================================================

#=======================================================================
class TestCookbook( MplTestCase ):
   """Cookbook unit test class."""

   # Uncomment any appropriate tags
   tags = [
            # 'gui',        # requires the creation of a gui window
            # 'agg',        # uses agg in the backend
            # 'agg-only',   # uses only agg in the backend
            # 'wx',         # uses wx in the backend
            # 'qt',         # uses qt in the backend
            # 'ps',         # uses the postscript backend
            # 'units',      # uses units in the test
            # 'PIL',        # uses PIL for image comparison
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
   def test_is_string_like( self ):
      """Test the 'is_string_like cookbook' function."""
      y = npy.arange( 10 )
      self.failUnless( cbook.is_string_like( y ) == False )
      y.shape = 10, 1
      self.failUnless( cbook.is_string_like( y ) == False )
      y.shape = 1, 10
      self.failUnless( cbook.is_string_like( y ) == False )


      self.failUnless( cbook.is_string_like( "hello world" ) )
      self.failUnless( cbook.is_string_like(10) == False )

   #--------------------------------------------------------------------
   #TODO: More cookbook tests

