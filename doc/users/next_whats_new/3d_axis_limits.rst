Setting 3D axis limits now set the limits exactly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, setting the limits of a 3D axis would always add a small margin to
the limits. Limits are now set exactly by default. The newly introduced rcparam
``axes3d.automargin`` can be used to revert to the old behavior where margin is
automatically added.

.. plot::
    :include-source: true
    :alt: Example of the new behavior of 3D axis limits, and how setting the rcparam reverts to the old behavior.

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    plt.rcParams['axes3d.automargin'] = True
    axs[0].set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title='Old Behavior')

    plt.rcParams['axes3d.automargin'] = False  # the default in 3.9.0
    axs[1].set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title='New Behavior')
