Default behavior of log scales changed to mask <= 0 values
``````````````````````````````````````````````````````````

Calling `matplotlib.axes.Axes.set_xscale` or `matplotlib.axes.Axes.set_yscale`
now uses 'mask' as the default method to handle invalid values (as opposed to
'clip'). This means that any values <= 0 on a log scale will not be shown.

Previously they were clipped to a very small number and shown.
