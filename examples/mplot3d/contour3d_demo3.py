from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = axes3d.Axes3D(fig)
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40)
cset = ax.contour(X, Y, Z, zdir='y', offset=40)

ax.set_xlabel('X')
ax.set_xlim3d(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim3d(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim3d(-100, 100)

plt.show()

