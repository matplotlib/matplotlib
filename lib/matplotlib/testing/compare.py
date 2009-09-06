#=======================================================================
""" A set of utilities for comparing results.
"""
#=======================================================================

import math
import operator
import os
import numpy as np
import shutil

#=======================================================================

__all__ = [
            'compare_float',
            'compare_images',
          ]

#-----------------------------------------------------------------------
def compare_float( expected, actual, relTol = None, absTol = None ):
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
def compare_images( expected, actual, tol, in_decorator=False ):
   '''Compare two image files - not the greatest, but fast and good enough.

   = EXAMPLE

   # img1 = "./baseline/plot.png"
   # img2 = "./output/plot.png"
   #            
   # compare_images( img1, img2, 0.001 ):

   = INPUT VARIABLES
   - expected  The filename of the expected image.
   - actual    The filename of the actual image.
   - tol       The tolerance (a unitless float).  This is used to
               determinte the 'fuzziness' to use when comparing images.
   - in_decorator If called from image_comparison decorator, this should be
               True. (default=False)
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

   diff_image = os.path.join(os.path.dirname(actual),
                             'failed-diff-'+os.path.basename(actual))
   save_diff_image( expected, actual, diff_image )

   if in_decorator:
      shutil.copyfile( expected, 'expected-'+os.path.basename(actual))
      results = dict(
         rms = rms,
         expected = str(expected),
         actual = str(actual),
         diff = str(diff_image),
         )
      return results
   else:
      # old-style call from mplTest directory
      msg = "  Error: Image files did not match.\n"       \
            "  RMS Value: " + str( rms / 10000.0 ) + "\n" \
            "  Expected:\n    " + str( expected ) + "\n"  \
            "  Actual:\n    " + str( actual ) + "\n"      \
            "  Difference:\n    " + str( diff_image ) + "\n"      \
            "  Tolerance: " + str( tol ) + "\n"
      return msg

def save_diff_image( expected, actual, output ):
   from PIL import Image
   expectedImage = np.array(Image.open( expected ).convert("RGB")).astype(np.float)
   actualImage = np.array(Image.open( actual ).convert("RGB")).astype(np.float)
   assert expectedImage.ndim==expectedImage.ndim
   assert expectedImage.shape==expectedImage.shape
   absDiffImage = abs(expectedImage-actualImage)
   # expand differences in luminance domain
   absDiffImage *= 10
   save_image_np = np.clip(absDiffImage,0,255).astype(np.uint8)
   save_image = Image.fromarray(save_image_np)
   save_image.save(output)
