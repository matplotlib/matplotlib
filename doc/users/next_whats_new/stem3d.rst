Stem plots in 3D Axes
---------------------

Stem plots are now supported on 3D Axes. Much like 2D stems,
`~.axes3d.Axes3D.stem3D` supports plotting the stems in various orientations:

.. plot::

    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta - np.pi/2)
    y = np.sin(theta - np.pi/2)
    z = theta
    directions = ['z', 'x', 'y']
    names = [r'$\theta$', r'$\cos\theta$', r'$\sin\theta$']

    fig, axs = plt.subplots(1, 3, figsize=(8, 4),
                            constrained_layout=True,
                            subplot_kw={'projection': '3d'})
    for ax, zdir, name in zip(axs, directions, names):
        ax.stem(x, y, z, orientation=zdir)
        ax.set_title(name)
    fig.suptitle(r'A parametric circle: $(x, y) = (\cos\theta, \sin\theta)$')

See also the :doc:`/gallery/mplot3d/stem3d_demo` demo.
