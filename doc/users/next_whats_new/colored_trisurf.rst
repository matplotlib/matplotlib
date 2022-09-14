3D Trisurf plots can now have colors and allow color-maps
---------------------------------------------------------

We can now show custom properties on triangle surface in 3D color coded.
You can provide an array of one value per vertex OR one value per face.
The faces will then be painted based on a color map. If one value per vertex
is specified (this is probably the default case, as we usually know data only on
the vertices) the vertex values will be averaged over the face an then color-mapped.

Now also shading of colored trisurfs is added. Before it was only possible for uniformly
colored meshes. This might not be what you want on colormaps, so the default is off.

.. plot::
    :include-source: true

    import numpy as np
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # Create a parametric sphere
    r = np.linspace(0, np.pi, 50)
    phi = np.linspace(-np.pi, np.pi, 50)
    r, phi = np.meshgrid(r, phi)
    r, phi = r.flatten(), phi.flatten()
    tri = mtri.Triangulation(r, phi)

    x = np.sin(r)*np.cos(phi)
    y = np.sin(r)*np.sin(phi)
    z = np.cos(r)

    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, C=phi, cmap=plt.cm.get_cmap('terrain'))
    plt.show()
