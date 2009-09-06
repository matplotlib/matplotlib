#=======================================================================
"""Defines the base matplotlib test-case."""
#=======================================================================

import os
import os.path
import unittest

import matplotlib.testing.compare as compare
import path_utils

#=======================================================================

__all__ = [ 'MplTestCase' ]

#=======================================================================
class MplTestCase( unittest.TestCase ):
   """This is the base class for the matplotlib unit-tests.

   It provides a few utility functions for accessing managed directories:
   - inputs   - All input files for the test case are stored here.
   - outputs  - All output files for the test case are written here.
   - baseline - All baseline files (those used for verifying results) for
                athe test case are stored here.
   """
   #--------------------------------------------------------------------
   def inFile( self, fname ):
      """Returns the pathname of the specified input file."""
      return os.path.join( self.inputDir, fname )

   def outFile( self, fname ):
      """Returns the pathname of the specified output file."""
      return os.path.join( self.outputDir, fname )

   def baseFile( self, fname ):
      """Returns the pathname of the specified basline file."""
      return os.path.join( self.baselineDir, fname )

   #--------------------------------------------------------------------
   def checkImage( self, outfname, tol = 1.0e-3, msg = "" ):
      """Check to see if the image is similair to one stored in the
         baseline directory.
      """
      if self.outputDir in outfname:
         # We are passed the path name and just want the file name.
         actualImage = outfname
         basename = path_utils.name( outfname )
      else:
         basename = outfname
         actualImage = self.outFile( basename )

      baselineImage = self.baseFile( basename )

      errorMessage = compare.compareImages( baselineImage, actualImage, tol )

      if errorMessage:
         self.fail( msg + "\n" + errorMessage )

   #--------------------------------------------------------------------
   def checkEq( expected, actual, msg = "" ):
      """Fail if the values are not equal, with the given message."""
      if not expected == actual:
         expectedStr = str( expected )
         actualStr = str( actual )
         isMultiLine = ( "\n" in expectedStr or "\n" in actualStr or
                        len( expectedStr ) > 70 or len( actualStr ) > 70 )

         if isMultiLine:
            if msg:
               msg += "\n\n"
            msg += "Expected:\n"
            msg += expectedStr + "\n\n"
            msg += "Actual:\n"
            msg += actualStr + "\n"
         else:
            if msg:
               msg += "\n"
            msg += "  Expected: " + expectedStr + "\n"
            msg += "  Actual:   " + actualStr + "\n"

         self.fail( msg )

   #--------------------------------------------------------------------
   def checkNeq( expected, actual, msg = "" ):
      """Fail is the values are equal, with the given message."""
      if expected == actual:
         expectedStr = str( expected )
         isMultiLine = ( "\n" in expectedStr or len( expectedStr ) > 55 )

         if isMultiLine:
            if msg:
               msg += "\n\n"
            msg += "Expected and actual should not be equal.\n"
            msg += "Expected and actual:\n"
            msg += expectedStr + "\n"
         else:
            if msg:
               msg += "\n"
            msg += "  Expected and actual should not be equal.\n"
            msg += "  Expected and actual: " + expectedStr + "\n"

         self.fail( msg )

   #--------------------------------------------------------------------
   def checkClose( expected, actual, relTol = None, absTol = None, msg = "" ):
      """Fail if the floating point values are not close enough, with
         the givem message.

      You can specify a relative tolerance, absolute tolerance, or both.
      """
      errorMessage = compare.compareFloat( expected, actual, relTol, absTol )

      if errorMessage:
         self.fail( msg + "\n" + errorMessage )

   #--------------------------------------------------------------------

