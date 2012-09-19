#=======================================================================

""" A set of utilities for comparing results.
"""
#=======================================================================

from __future__ import division

import matplotlib
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing import image_util, util
from matplotlib import _png
from matplotlib import _get_configdir
from distutils import version
import hashlib
import math
import operator
import os
import numpy as np
import shutil
import subprocess
import sys
from functools import reduce

#=======================================================================

__all__ = [
            'compare_float',
            'compare_images',
            'comparable_formats',
          ]

#-----------------------------------------------------------------------

def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting `purpose` before the file's
    extension.
    """
    base, ext = os.path.splitext(fname)
    return '%s-%s%s' % (base, purpose, ext)

def compare_float( expected, actual, relTol = None, absTol = None ):
   """Fail if the floating point values are not close enough, with
      the givem message.

   You can specify a relative tolerance, absolute tolerance, or both.
   """
   if relTol is None and absTol is None:
      exMsg = "You haven't specified a 'relTol' relative tolerance "
      exMsg += "or a 'absTol' absolute tolerance function argument.  "
      exMsg += "You must specify one."
      raise ValueError(exMsg)

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
def get_cache_dir():
   cache_dir = os.path.join(_get_configdir(), 'test_cache')
   if not os.path.exists(cache_dir):
      try:
         os.makedirs(cache_dir)
      except IOError:
         return None
   if not os.access(cache_dir, os.W_OK):
      return None
   return cache_dir

def get_file_hash(path, block_size=2**20):
   md5 = hashlib.md5()
   with open(path, 'rb') as fd:
      while True:
         data = fd.read(block_size)
         if not data:
            break
         md5.update(data)
   return md5.hexdigest()

converter = { }

def make_external_conversion_command(cmd):
   def convert(old, new):
      cmdline = cmd(old, new)
      pipe = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = pipe.communicate()
      errcode = pipe.wait()
      if not os.path.exists(new) or errcode:
         msg = "Conversion command failed:\n%s\n" % ' '.join(cmdline)
         if stdout:
            msg += "Standard output:\n%s\n" % stdout
         if stderr:
            msg += "Standard error:\n%s\n" % stderr
         raise IOError(msg)

   return convert

if matplotlib.checkdep_ghostscript() is not None:
    def make_ghostscript_conversion_command():
        # FIXME: make checkdep_ghostscript return the command
        if sys.platform == 'win32':
            gs = 'gswin32c'
        else:
            gs = 'gs'
        cmd = [gs, '-q', '-sDEVICE=png16m', '-sOutputFile=-']

        process = util.MiniExpect(cmd)

        def do_convert(old, new):
            process.expect("GS>")
            process.sendline("(%s) run" % old.replace('\\', '/'))
            with open(new, 'wb') as fd:
                process.expect(">>showpage, press <return> to continue<<", fd)
            process.sendline('')

        return do_convert

    converter['pdf'] = make_ghostscript_conversion_command()
    converter['eps'] = make_ghostscript_conversion_command()


if matplotlib.checkdep_inkscape() is not None:
   cmd = lambda old, new: \
             ['inkscape', '-z', old, '--export-png', new]
   converter['svg'] = make_external_conversion_command(cmd)

def comparable_formats():
   '''Returns the list of file formats that compare_images can compare
   on this system.'''
   return ['png'] + converter.keys()

def convert(filename, cache):
   '''
   Convert the named file into a png file.  Returns the name of the
   created file.

   If *cache* is True, the result of the conversion is cached in
   `~/.matplotlib/test_cache/`.  The caching is based on a hash of the
   exact contents of the input file.  The is no limit on the size of
   the cache, so it may need to be manually cleared periodically.
   '''
   base, extension = filename.rsplit('.', 1)
   if extension not in converter:
      raise ImageComparisonFailure("Don't know how to convert %s files to png" % extension)
   newname = base + '_' + extension + '.png'
   if not os.path.exists(filename):
      raise IOError("'%s' does not exist" % filename)

   # Only convert the file if the destination doesn't already exist or
   # is out of date.
   if (not os.path.exists(newname) or
       os.stat(newname).st_mtime < os.stat(filename).st_mtime):
      if cache:
         cache_dir = get_cache_dir()
      else:
         cache_dir = None

      if cache_dir is not None:
         hash = get_file_hash(filename)
         new_ext = os.path.splitext(newname)[1]
         cached_file = os.path.join(cache_dir, hash + new_ext)
         if os.path.exists(cached_file):
            shutil.copyfile(cached_file, newname)
            return newname

      converter[extension](filename, newname)

      if cache_dir is not None:
         shutil.copyfile(newname, cached_file)

   return newname

verifiers = { }

def verify(filename):
   """
   Verify the file through some sort of verification tool.
   """
   if not os.path.exists(filename):
      raise IOError("'%s' does not exist" % filename)
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
         raise IOError(msg)

# Turning this off, because it seems to cause multiprocessing issues
if matplotlib.checkdep_xmllint() and False:
   verifiers['svg'] = lambda filename: [
      'xmllint', '--valid', '--nowarning', '--noout', filename]

def crop_to_same(actual_path, actual_image, expected_path, expected_image):
   # clip the images to the same size -- this is useful only when
   # comparing eps to pdf
   if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
      aw, ah = actual_image.shape
      ew, eh = expected_image.shape
      actual_image = actual_image[int(aw/2-ew/2):int(aw/2+ew/2),int(ah/2-eh/2):int(ah/2+eh/2)]
   return actual_image, expected_image

def calculate_rms(expectedImage, actualImage):
   # compare the resulting image histogram functions
   expected_version = version.LooseVersion("1.6")
   found_version = version.LooseVersion(np.__version__)

   # On Numpy 1.6, we can use bincount with minlength, which is much faster than
   # using histogram
   if found_version >= expected_version:
      rms = 0

      for i in xrange(0, 3):
         h1p = expectedImage[:,:,i]
         h2p = actualImage[:,:,i]

         h1h = np.bincount(h1p.ravel(), minlength=256)
         h2h = np.bincount(h2p.ravel(), minlength=256)

         rms += np.sum(np.power((h1h-h2h), 2))
   else:
      rms = 0
      bins = np.arange(257)

      for i in xrange(0, 3):
         h1p = expectedImage[:,:,i]
         h2p = actualImage[:,:,i]

         h1h = np.histogram(h1p, bins=bins)[0]
         h2h = np.histogram(h2p, bins=bins)[0]

         rms += np.sum(np.power((h1h-h2h), 2))

   rms = np.sqrt(rms / (256 * 3))

   return rms


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

   verify(actual)

   # Convert the image to png
   extension = expected.split('.')[-1]
   if extension != 'png':
      actual = convert(actual, False)
      expected = convert(expected, True)

   # open the image files and remove the alpha channel (if it exists)
   expectedImage = _png.read_png_int( expected )
   actualImage = _png.read_png_int( actual )

   actualImage, expectedImage = crop_to_same(actual, actualImage, expected, expectedImage)

   # compare the resulting image histogram functions
   expected_version = version.LooseVersion("1.6")
   found_version = version.LooseVersion(np.__version__)

   rms = calculate_rms(expectedImage, actualImage)

   diff_image = make_test_filename(actual, 'failed-diff')

   if ( (rms / 10000.0) <= tol ):
      if os.path.exists(diff_image):
         os.unlink(diff_image)
      return None

   # For Agg-rendered images, we can retry by ignoring pixels with
   # differences of only 1
   if extension == 'png':
       # Remove differences of only 1
       diffImage = np.abs(np.asarray(actualImage, dtype=np.int) -
                          np.asarray(expectedImage, dtype=np.int))
       actualImage = np.where(diffImage <= 1, expectedImage, actualImage)

       rms = calculate_rms(expectedImage, actualImage)

       if ( (rms / 10000.0) <= tol ):
           if os.path.exists(diff_image):
               os.unlink(diff_image)
           return None

   save_diff_image( expected, actual, diff_image )

   if in_decorator:
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
   expectedImage = _png.read_png( expected )
   actualImage = _png.read_png( actual )
   actualImage, expectedImage = crop_to_same(actual, actualImage, expected, expectedImage)
   expectedImage = np.array(expectedImage).astype(np.float)
   actualImage = np.array(actualImage).astype(np.float)
   assert expectedImage.ndim==actualImage.ndim
   assert expectedImage.shape==actualImage.shape
   absDiffImage = abs(expectedImage-actualImage)

   # expand differences in luminance domain
   absDiffImage *= 255 * 10
   save_image_np = np.clip(absDiffImage, 0, 255).astype(np.uint8)
   height, width, depth = save_image_np.shape

   # The PDF renderer doesn't produce an alpha channel, but the
   # matplotlib PNG writer requires one, so expand the array
   if depth == 3:
      with_alpha = np.empty((height, width, 4), dtype=np.uint8)
      with_alpha[:,:,0:3] = save_image_np
      save_image_np = with_alpha

   # Hard-code the alpha channel to fully solid
   save_image_np[:,:,3] = 255

   _png.write_png(save_image_np.tostring(), width, height, output)
