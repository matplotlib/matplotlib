`~.axes.Axes.set_title` gains a y keyword argument to control auto positioning
------------------------------------------------------------------------------
`~.axes.Axes.set_title` tries to auto-position the title to avoid any
decorators on the top x-axis.  This is not always desirable so now
*y* is an explicit keyword argument of `~.axes.Axes.set_title`.  It
defaults to *None* which means to use auto-positioning.  If a value is
supplied (i.e. the pre-3.0 default was ``y=1.0``) then auto-positioning is
turned off.  This can also be set with the new rcParameter :rc:`axes.titley`.  
