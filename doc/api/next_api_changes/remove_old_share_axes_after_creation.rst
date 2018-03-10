Remove share through `matplotlib.Axes.get_shared_{x,y,z}_axes`
--------------------------------------------------------------

Previously when different axes are created with different parent/master axes,
the share would still be symmetric and transitive if an unconventional
method through `matplotlib.Axes.get_shared_x_axes`
is used to share the axes after creation. With the new sharing mechanism
this is no longer possible.
