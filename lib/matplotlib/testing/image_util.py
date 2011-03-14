# This module contains some functionality from the Python Imaging
# Library, that has been ported to use Numpy arrays rather than PIL
# Image objects.


# The Python Imaging Library is

# Copyright (c) 1997-2009 by Secret Labs AB
# Copyright (c) 1995-2009 by Fredrik Lundh

# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:

# Permission to use, copy, modify, and distribute this software and its
# associated documentation for any purpose and without fee is hereby
# granted, provided that the above copyright notice appears in all
# copies, and that both that copyright notice and this permission notice
# appear in supporting documentation, and that the name of Secret Labs
# AB or the author not be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior
# permission.

# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO
# THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import print_function
import numpy as np

# TODO: Vectorize this
def autocontrast(image, cutoff=0):
    """
    Maximize image contrast, based on histogram.  This completely
    ignores the alpha channel.
    """
    assert image.dtype == np.uint8

    output_image = np.empty((image.shape[0], image.shape[1], 3), np.uint8)

    for i in xrange(0, 3):
        plane = image[:,:,i]
        output_plane = output_image[:,:,i]
        h = np.histogram(plane, bins=256)[0]
        if cutoff:
            # cut off pixels from both ends of the histogram
            # get number of pixels
            n = 0
            for ix in xrange(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff / 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] = h[lo] - cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the hi end
            cut = n * cutoff / 100
            for hi in xrange(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] = h[hi] - cut
                    cut = 0
                if cut <= 0:
                    break

        # find lowest/highest samples after preprocessing
        for lo in xrange(256):
            if h[lo]:
                break
        for hi in xrange(255, -1, -1):
            if h[hi]:
                break

        if hi <= lo:
            output_plane[:,:] = plane
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            lut = np.arange(256, dtype=np.float)
            lut *= scale
            lut += offset
            lut = lut.clip(0, 255)
            lut = lut.astype(np.uint8)

            output_plane[:,:] = lut[plane]

    return output_image
