URL Support for Images in PDF Backend
-------------------------------------

The PDF backend can now generate clickable images if a URL is provided to the
image. There are a few limitations worth noting though:

* If parts of the image are clipped, the non-visible parts are still clickable.
* If there are transforms applied to the image, the whole enclosing rectangle
  is clickable. However, if you use ``interpolation='none'`` for the image,
  only the transformed image area is clickable (depending on viewer support).
