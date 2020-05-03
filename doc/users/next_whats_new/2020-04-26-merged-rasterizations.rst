Consecutive rasterized draws now merged
---------------------------------------

Elements of a vector output can be individually set to rasterized, using
the ``rasterized`` keyword, or `~.artist.Artist.set_rasterized()`. This can
be useful to reduce file sizes. For figures with multiple raster elements
they are now automatically merged into a smaller number of bitmaps where
this will not effect the visual output. For cases with many elements this
can result in significantly smaller file sizes.

To ensure this happens do not place vector elements between raster ones.

To inhibit this merging set ``Figure.suppressComposite`` to True.
