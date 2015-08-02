Zero r/cstride support in plot_wireframe
----------------------------------------

Adam Hughes added support to mplot3d's plot_wireframe to draw only row or
column line plots. 


Example::

    from mpl_toolkits.mplot3d import Axes3D, axes3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=0)
