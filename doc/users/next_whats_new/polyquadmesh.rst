``PolyQuadMesh`` is a new class for drawing quadrilateral meshes
----------------------------------------------------------------

`~.Axes.pcolor` previously returned a flattened `.PolyCollection` with only
the valid polygons (unmasked) contained within it. Now, we return a `.PolyQuadMesh`,
which is a mixin incorporating the usefulness of 2D array and mesh coordinates
handling, but still inheriting the draw methods of `.PolyCollection`, which enables
more control over the rendering properties than a normal `.QuadMesh` that is
returned from `~.Axes.pcolormesh`. The new class subclasses `.PolyCollection` and thus
should still behave the same as before. This new class keeps track of the mask for
the user and updates the Polygons that are sent to the renderer appropriately.

.. plot::

    arr = np.arange(12).reshape((3, 4))

    fig, ax = plt.subplots()
    pc = ax.pcolor(arr)

    # Mask one element and show that the hatch is also not drawn
    # over that region
    pc.set_array(np.ma.masked_equal(arr, 5))
    pc.set_hatch('//')

    plt.show()
