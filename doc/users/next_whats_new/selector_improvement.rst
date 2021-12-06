Selectors improvement: rotation, aspect ratio correction and add/remove state
-----------------------------------------------------------------------------

The `~matplotlib.widgets.RectangleSelector` and
`~matplotlib.widgets.EllipseSelector` can now be rotated interactively between
-45° and 45°. The range limits are currently dictated by the implementation.
The rotation is enabled or disabled by striking the *r* key
('r' is the default key mapped to 'rotate' in *state_modifier_keys*) or by calling
``selector.add_state('rotate')``.

The aspect ratio of the axes can now be taken into account when using the
"square" state. This is enabled by specifying ``use_data_coordinates='True'`` when
the selector is initialized.

In addition to changing selector state interactively using the modifier keys
defined in *state_modifier_keys*, the selector state can now be changed
programmatically using the *add_state* and *remove_state* methods.


.. code-block:: python

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import numpy as np

    values = np.arange(0, 100)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(values, values)

    selector = RectangleSelector(ax, print, interactive=True,
                                 drag_from_anywhere=True,
                                 use_data_coordinates=True)
    selector.add_state('rotate') # alternatively press 'r' key
    # rotate the selector interactively

    selector.remove_state('rotate') # alternatively press 'r' key

    selector.add_state('square')
