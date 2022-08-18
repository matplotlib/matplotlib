Make the resample method on `.Colormap` instances public
--------------------------------------------------------

On `.LinearSegmentedColormap` and `.ListedColormap` the previously private
``_resample`` method is made public as `.Colormap.resampled`.  This method
creates a new `.Colormap` instance with the specified lookup table size.
