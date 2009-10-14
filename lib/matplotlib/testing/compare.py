#=======================================================================
""" A set of utilities for comparing results.
"""
#=======================================================================

import matplotlib
from matplotlib.testing.noseclasses import ImageComparisonFailure
import math
import operator
import os
import numpy as np
import shutil
import subprocess
import sys

#=======================================================================

__all__ = [
            'compare_float',
            'compare_images',
            'comparable_formats',
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
# A dictionary that maps filename extensions to functions that map
# parameters old and new to a list that can be passed to Popen to
# convert files with that extension to png format.
converter = { }

if matplotlib.checkdep_ghostscript() is not None:
   # FIXME: make checkdep_ghostscript return the command
   if sys.platform == 'win32':
      gs = 'gswin32c'
   else:
      gs = 'gs'
   cmd = lambda old, new: \
       [gs, '-q', '-sDEVICE=png16m', '-dNOPAUSE', '-dBATCH',
        '-sOutputFile=' + new, old]
   converter['pdf'] = cmd
   converter['eps'] = cmd

if matplotlib.checkdep_inkscape() is not None:
   cmd = lambda old, new: \
       ['inkscape', old, '--export-png=' + new]
   converter['svg'] = cmd

def comparable_formats():
   '''Returns the list of file formats that compare_images can compare
   on this system.'''
   return ['png'] + converter.keys()

def convert(filename):
   '''
   Convert the named file into a png file.
   Returns the name of the created file.
   '''
   base, extension = filename.rsplit('.', 1)
   if extension not in converter:
      raise ImageComparisonFailure, "Don't know how to convert %s files to png" % extension
   newname = base + '_' + extension + '.png'
   if not os.path.exists(filename):
      raise IOError, "'%s' does not exist" % filename
   cmd = converter[extension](filename, newname)
   pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   stdout, stderr = pipe.communicate()
   errcode = pipe.wait()
   if not os.path.exists(newname) or errcode:
      msg = "Conversion command failed:\n%s\n" % ' '.join(cmd)
      if stdout:
         msg += "Standard output:\n%s\n" % stdout
      if stderr:
         msg += "Standard error:\n%s\n" % stderr
      raise IOError, msg
   return newname

verifiers = { }

def verify(filename):
   """
   Verify the file through some sort of verification tool.
   """
   if not os.path.exists(filename):
      raise IOError, "'%s' does not exist" % filename
   base, extension = filename.rsplit('.', 1)
   verifier = verifiers.get(extension, None)
   if verifier is not None:
      cmd = verifier(filename)
      pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = pipe.communicate()
      errcode = pipe.wait()
      if errcode != 0:
         msg = "File verification command failed:\n%s\n" % ' '.join(cmd)
         if stdout:
            msg += "Standard output:\n%s\n" % stdout
         if stderr:
            msg += "Standard error:\n%s\n" % stderr
         raise IOError, msg

# Turning this off, because it seems to cause multiprocessing issues
if matplotlib.checkdep_xmllint() and False:
   verifiers['svg'] = lambda filename: [
      'xmllint', '--valid', '--nowarning', '--noout', filename]

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
               determine the 'fuzziness' to use when comparing images.
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
      if in_decorator:
         raise NotImplementedError, e
      else:
         return msg

   verify(actual)

   # Convert the image to png
   extension = expected.split('.')[-1]
   if extension != 'png':
      actual, expected = convert(actual), convert(expected)

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

   diff_image = os.path.join(os.path.dirname(actual),
                             'failed-diff-'+os.path.basename(actual))
   expected_copy = 'expected-'+os.path.basename(actual)

   if ( (rms / 10000.0) <= tol ):
      if os.path.exists(diff_image):
         os.unlink(diff_image)
      if os.path.exists(expected_copy):
         os.unlink(expected_copy)
      return None

   save_diff_image( expected, actual, diff_image )

   if in_decorator:
      shutil.copyfile( expected, expected_copy )
      results = dict(
         rms = rms,
         expected = str(expected),
         actual = str(actual),
         diff = str(diff_image),
         )
      return results
   else:
      # expected_copy is only for in_decorator case
      if os.path.exists(expected_copy):
         os.unlink(expected_copy)
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
