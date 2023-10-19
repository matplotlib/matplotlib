Changes for 0.71
================

.. code-block:: text

   Significant numerix namespace changes, introduced to resolve
   namespace clashes between python built-ins and mlab names.
   Refactored numerix to maintain separate modules, rather than
   folding all these names into a single namespace.  See the following
   mailing list threads for more information and background

     http://sourceforge.net/mailarchive/forum.php?thread_id=6398890&forum_id=36187
     http://sourceforge.net/mailarchive/forum.php?thread_id=6323208&forum_id=36187


   OLD usage

     from matplotlib.numerix import array, mean, fft

   NEW usage

     from matplotlib.numerix import array
     from matplotlib.numerix.mlab import mean
     from matplotlib.numerix.fft import fft

   numerix dir structure mirrors numarray (though it is an incomplete
   implementation)

     numerix
     numerix/mlab
     numerix/linear_algebra
     numerix/fft
     numerix/random_array

   but of course you can use 'numerix : Numeric' and still get the
   symbols.

   pylab still imports most of the symbols from Numerix, MLab, fft,
   etc, but is more cautious.  For names that clash with python names
   (min, max, sum), pylab keeps the builtins and provides the numeric
   versions with an a* prefix, e.g., (amin, amax, asum)
