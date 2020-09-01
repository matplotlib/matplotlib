Consecutive rasterized draws now merged
---------------------------------------

Tracking of depth of raster draws has moved from
`.backend_mixed.MixedModeRenderer.start_rasterizing` and
`.backend_mixed.MixedModeRenderer.stop_rasterizing` into
`.artist.allow_rasterization`. This means the start and stop functions are
only called when the rasterization actually needs to be started and stopped.

The output of vector backends will change in the case that rasterized
elements are merged. This should not change the appearance of outputs.

The renders in 3rd party backends are now expected to have
``self._raster_depth`` and ``self._rasterizing`` initialized to ``0`` and
``False`` respectively.
