Touchscreen Support
-------------------

Support for touch-to-drag and pinch-to-zoom have been added for the
Qt4 and Qt5 backends. For other/custom backends, the interface in
`NavigationToolbar2` is general, so that the backends only need to
pass a list of the touch points, and `NavigationToolbar2` will do the rest.
Support is added separately for touch rotating and zooming in `Axes3D`.