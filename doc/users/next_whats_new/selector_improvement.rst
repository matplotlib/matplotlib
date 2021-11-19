Selectors improvement: rotation, aspect ratio correction and add/remove state
-----------------------------------------------------------------------------

The `~matplotlib.widgets.RectangleSelector` and
`~matplotlib.widgets.EllipseSelector` can now be rotated interactively.
The rotation is enabled or disable by striking the *r* key
(default value of 'rotate' in *state_modifier_keys*) or by calling
*selector.add_state('rotate')*.

The aspect ratio of the axes can now be taken into account when using the
"square" state. This can be enable or disable by striking the *d* key
(default value of 'data_coordinates' in *state_modifier_keys*)
or by calling *selector.add_state('rotate')*.

In addition to changing selector state interactively using the modifier keys
defined in *state_modifier_keys*, the selector state can now be changed
programmatically using the *add_state* and *remove_state* method.


.. code-block:: python

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import numpy as np

    values = np.arange(0, 100)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(values, values)

    selector = RectangleSelector(ax, print, interactive=True, drag_from_anywhere=True)
    selector.add_state('rotate') # alternatively press 'r' key
    # rotate the selector interactively

    selector.remove_state('rotate') # alternatively press 'r' key

    selector.add_state('square')
    # to keep the aspect ratio in data coordinates
    selector.add_state('data_coordinates')
