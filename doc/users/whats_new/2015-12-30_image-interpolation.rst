Improved image support
----------------------

Prior to version 2.0, matplotlib resampled images by first applying
the color map and then resizing the result.  Since the resampling was
performed on the colored image, this introduced colors in the output
image that didn't actually exist in the color map.  Now, images are
resampled first (and entirely in floating-point, if the input image is
floating-point), and then the color map is applied.

In order to make this important change, the image handling code was
almost entirely rewritten.  As a side effect, image resampling uses
less memory and fewer datatype conversions than before.

The experimental private feature where one could "skew" an image by
setting the private member ``_image_skew_coordinate`` has been
removed.  Instead, images will obey the transform of the axes on which
they are drawn.
