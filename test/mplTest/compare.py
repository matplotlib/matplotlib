#=======================================================================
""" A set of utilities for comparing results.
"""
#=======================================================================

import math
import operator

#=======================================================================

__all__ = [
            'compareFloat',
            'compareImages',
          ]

#-----------------------------------------------------------------------
def compareFloat( expected, actual, relTol = None, absTol = None ):
   """Fail if the floating point values are not close enough, with
      the givem message.

   You can specify a relative tolerance, absolute tolerance, or both.
   """
   if relTol is None and absTol is None:
      exMsg = "You haven't specified a 'relTol' relative tolerance "
      exMsg += "or a 'absTol' absolute tolerance function argument.  "
      exMsg += "You must specify one."
      raise ValueError, exMsg

   msg = ""

   if absTol is not None:
      absDiff = abs( expected - actual )
      if absTol < absDiff:
         expectedStr = str( expected )
         actualStr = str( actual )
         absDiffStr = str( absDiff )
         absTolStr = str( absTol )

         msg += "\n"
         msg += "  Expected: " + expectedStr + "\n"
         msg += "  Actual:   " + actualStr + "\n"
         msg += "  Abs Diff: " + absDiffStr + "\n"
         msg += "  Abs Tol:  " + absTolStr + "\n"

   if relTol is not None:
      # The relative difference of the two values.  If the expected value is
      # zero, then return the absolute value of the difference.
      relDiff = abs( expected - actual )
      if expected:
         relDiff = relDiff / abs( expected )

      if relTol < relDiff:

         # The relative difference is a ratio, so it's always unitless.
         relDiffStr = str( relDiff )
         relTolStr = str( relTol )

         expectedStr = str( expected )
         actualStr = str( actual )

         msg += "\n"
         msg += "  Expected: " + expectedStr + "\n"
         msg += "  Actual:   " + actualStr + "\n"
         msg += "  Rel Diff: " + relDiffStr + "\n"
         msg += "  Rel Tol:  " + relTolStr + "\n"

   if msg:
      return msg
   else:
      return None

#-----------------------------------------------------------------------
def compareImages( expected, actual, tol ):
   '''Compare two image files - not the greatest, but fast and good enough.

   = EXAMPLE

   # img1 = "./baseline/plot.png"
   # img2 = "./output/plot.png"
   #            
   # compareImage( img1, img2, 0.001 ):

   = INPUT VARIABLES
   - expected  The filename of the expected image.
   - actual    The filename of the actual image.
   - tol       The tolerance (a unitless float).  This is used to
               determinte the 'fuzziness' to use when comparing images.
   '''

   try:
      from PIL import Image, ImageOps, ImageFilter
   except ImportError, e:
      msg = "Image Comparison requires the Python Imaging Library to " \
            "be installed.  To run tests without using PIL, then use " \
            "the '--without-tag=PIL' command-line option.\n"           \
            "Importing PIL failed with the following error:\n%s" % e
      return msg

   # open the image files and remove the alpha channel (if it exists)
   expectedImage = Image.open( expected ).convert("RGB")
   actualImage = Image.open( actual ).convert("RGB")

   # normalize the images
   expectedImage = ImageOps.autocontrast( expectedImage, 2 )
   actualImage = ImageOps.autocontrast( actualImage, 2 )

   # compare the resulting image histogram functions
   h1 = expectedImage.histogram()
   h2 = actualImage.histogram()
   rms = math.sqrt( reduce(operator.add, map(lambda a,b: (a-b)**2, h1, h2)) / len(h1) )

   if ( (rms / 10000.0) <= tol ):
      return None
   else:
      msg = "  Error: Image files did not match.\n"       \
            "  RMS Value: " + str( rms / 10000.0 ) + "\n" \
            "  Expected:\n    " + str( expected ) + "\n"  \
            "  Actual:\n    " + str( actual ) + "\n"      \
            "  Tolerance: " + str( tol ) + "\n"
      return msg

