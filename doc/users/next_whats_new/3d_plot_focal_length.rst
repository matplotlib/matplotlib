Give the 3D camera a custom focal length
----------------------------------------

Users can now better mimic real-world cameras by specifying the focal length of
the virtual camera in 3D plots. The default focal length of 1 corresponds to a
Field of View (FOV) of 90 deg, and is backwards-compatible with existing 3D
plots. An increased focal length between 1 and infinity "flattens" the image,
while a decreased focal length between 1 and 0 exaggerates the perspective and
gives the image more apparent depth.

The focal length can be calculated from a desired FOV via the equation:

.. mathmpl::

    focal\_length = 1/\tan(FOV/2)

.. plot::
    :include-source: true

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from numpy import inf
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
    X, Y, Z = axes3d.get_test_data(0.05)
    focal_lengths = [0.2, 1, inf]
    for ax, fl in zip(axs, focal_lengths):
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        ax.set_proj_type('persp', focal_length=fl)
        ax.set_title(f"focal_length = {fl}")
    fig.set_size_inches(10, 4)
    plt.show()
