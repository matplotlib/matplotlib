Image interpolation now possible at RGBA stage
----------------------------------------------

Images in Matplotlib created via `~.axes.Axes.imshow` are resampled to match 
the resolution of the current canvas.  It is useful to apply an anto-aliasing
filter when downsampling to reduce Moire effects.  By default, interpolation 
is done on the data, a norm applied, and then the colormapping performed. 

However, it is often desireable for the anti-aliasing interpolation to happen 
in RGBA space, where the colors are interpolated rather than the data.  This 
usually leads to colors outside the colormap, but visually blends adjacent 
colors, and is what browsers and other image processing software does. 

A new keyword argument *interpolation_stage* is provided for 
`~.axes.Axes.imshow` to set the stage at which the anti-aliasing interpolation 
happens.  The default is the current behaviour of "data", with the alternative
being "rgba" for the newly-available behavior.  

For more details see the discussion of the new keyword argument in
:doc:`/gallery/images_contours_and_fields/image_antialiasing`.

