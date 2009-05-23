import random
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.colors import Normalize, colorConverter

def test_scatter():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    n = 100
    for c,zl,zh in [('r',-50,-25),('b',-30,-5)]:
        xs,ys,zs = zip(*
                       [(random.randrange(23,32),
                         random.randrange(100),
                         random.randrange(zl,zh)
                         ) for i in range(n)])
        ax.scatter3D(xs,ys,zs, c=c)

    ax.set_xlabel('------------ X Label --------------------')
    ax.set_ylabel('------------ Y Label --------------------')
    ax.set_zlabel('------------ Z Label --------------------')

def test_wire():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    X,Y,Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X,Y,Z, rstride=10,cstride=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_surface():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    X,Y,Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X,Y,Z, rstride=10,cstride=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_contour():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    X,Y,Z = axes3d.get_test_data(0.05)
    cset = ax.contour3D(X,Y,Z)
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_contourf():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    X,Y,Z = axes3d.get_test_data(0.05)
    cset = ax.contourf3D(X,Y,Z)
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_plot():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    xs = np.arange(0,4*np.pi+0.1,0.1)
    ys = np.sin(xs)
    ax.plot(xs,ys, label='zl')
    ax.plot(xs,ys+max(xs),label='zh')
    ax.plot(xs,ys,dir='x', label='xl')
    ax.plot(xs,ys,dir='x', z=max(xs),label='xh')
    ax.plot(xs,ys,dir='y', label='yl')
    ax.plot(xs,ys,dir='y', z=max(xs), label='yh')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def test_polys():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

    xs = np.arange(0,10,0.4)
    verts = []
    zs = [0.0,1.0,2.0,3.0]
    for z in zs:
        ys = [random.random() for x in xs]
        ys[0],ys[-1] = 0,0
        verts.append(zip(xs,ys))

    from matplotlib.collections import PolyCollection
    poly = PolyCollection(verts, facecolors = [cc('r'),cc('g'),cc('b'),
                                               cc('y')])
    poly.set_alpha(0.7)
    ax.add_collection(poly,zs=zs,dir='y')

    ax.set_xlim(0,10)
    ax.set_ylim(-1,4)
    ax.set_zlim(0,1)

def test_scatter2D():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    xs = [random.random() for i in range(20)]
    ys = [random.random() for x in xs]
    ax.scatter(xs, ys)
    ax.scatter(xs, ys, dir='y', c='r')
    ax.scatter(xs, ys, dir='x', c='g')

def test_bar2D():
    f = plt.figure()
    ax = axes3d.Axes3D(f)

    for c,z in zip(['r','g','b', 'y'],[30,20,10,0]):
        xs = np.arange(20)
        ys = [random.random() for x in xs]
        ax.bar(xs, ys, z=z, dir='y', color=c, alpha=0.8)

if __name__ == "__main__":

    test_scatter()
    test_wire()
    test_surface()
    test_contour()
    test_contourf()
    test_plot()
    test_polys()
    test_scatter2D()
    test_bar2D()

    plt.show()
