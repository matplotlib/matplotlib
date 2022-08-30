3D plots gained a 3rd "roll" viewing angle
------------------------------------------

3D plots can now be viewed from any orientation with the addition of a 3rd roll
angle, which rotates the plot about the viewing axis. Interactive rotation
using the mouse still only controls elevation and azimuth, meaning that this
feature is relevant to users who create more complex camera angles
programmatically. The default roll angle of 0 is backwards-compatible with
existing 3D plots.

.. plot::
    :include-source: true

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    ax.view_init(elev=0, azim=0, roll=30)
    ax.set_title('elev=0, azim=0, roll=30')
    plt.show()
