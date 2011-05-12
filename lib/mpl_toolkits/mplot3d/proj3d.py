#!/usr/bin/python
# 3dproj.py
#
"""
Various transforms used for by the 3D code
"""

from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
import numpy.linalg as linalg



def line2d(p0, p1):
    """
    Return 2D equation of line in the form ax+by+c = 0
    """
    # x + x1  = 0
    x0, y0 = p0[:2]
    x1, y1 = p1[:2]
    #
    if x0 == x1:
        a = -1
        b = 0
        c = x1
    elif y0 == y1:
        a = 0
        b = 1
        c = -y1
    else:
        a = (y0-y1)
        b = (x0-x1)
        c = (x0*y1 - x1*y0)
    return a, b, c

def line2d_dist(l, p):
    """
    Distance from line to point
    line is a tuple of coefficients a,b,c
    """
    a, b, c = l
    x0, y0 = p
    return abs((a*x0 + b*y0 + c)/np.sqrt(a**2+b**2))


def line2d_seg_dist(p1, p2, p0):
    """distance(s) from line defined by p1 - p2 to point(s) p0

    p0[0] = x(s)
    p0[1] = y(s)

    intersection point p = p1 + u*(p2-p1)
    and intersection point lies within segment if u is between 0 and 1
    """

    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x01 = np.asarray(p0[0]) - p1[0]
    y01 = np.asarray(p0[1]) - p1[1]

    u = (x01*x21 + y01*y21)/float(abs(x21**2 + y21**2))
    u = np.clip(u, 0, 1)
    d = np.sqrt((x01 - u*x21)**2 + (y01 - u*y21)**2)

    return d

def test_lines_dists():
    import pylab
    ax = pylab.gca()

    xs, ys = (0,30), (20,150)
    pylab.plot(xs, ys)
    points = zip(xs, ys)
    p0, p1 = points

    xs, ys = (0,0,20,30), (100,150,30,200)
    pylab.scatter(xs, ys)

    dist = line2d_seg_dist(p0, p1, (xs[0], ys[0]))
    dist = line2d_seg_dist(p0, p1, np.array((xs, ys)))
    for x, y, d in zip(xs, ys, dist):
        c = Circle((x, y), d, fill=0)
        ax.add_patch(c)

    pylab.xlim(-200, 200)
    pylab.ylim(-200, 200)
    pylab.show()

def mod(v):
    """3d vector length"""
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax):
    dx, dy, dz = (xmax-xmin), (ymax-ymin), (zmax-zmin)
    return np.array([
        [1.0/dx,0,0,-xmin/dx],
        [0,1.0/dy,0,-ymin/dy],
        [0,0,1.0/dz,-zmin/dz],
        [0,0,0,1.0]])

def test_world():
    xmin, xmax = 100, 120
    ymin, ymax = -100, 100
    zmin, zmax = 0.1, 0.2
    M = world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    print M

def view_transformation(E, R, V):
    n = (E - R)
    ## new
#    n /= mod(n)
#    u = np.cross(V,n)
#    u /= mod(u)
#    v = np.cross(n,u)
#    Mr = np.diag([1.]*4)
#    Mt = np.diag([1.]*4)
#    Mr[:3,:3] = u,v,n
#    Mt[:3,-1] = -E
    ## end new

    ## old
    n = n / mod(n)
    u = np.cross(V, n)
    u = u / mod(u)
    v = np.cross(n, u)
    Mr = [[u[0],u[1],u[2],0],
          [v[0],v[1],v[2],0],
          [n[0],n[1],n[2],0],
          [0,   0,   0,   1],
          ]
    #
    Mt = [[1, 0, 0, -E[0]],
          [0, 1, 0, -E[1]],
          [0, 0, 1, -E[2]],
          [0, 0, 0, 1]]
    ## end old

    return np.dot(Mr, Mt)

def persp_transformation(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-1,0]
                     ])

def proj_transform_vec(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs

def proj_transform_vec_clip(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    tis = (vecw[0] >= 0) * (vecw[0] <= 1) * (vecw[1] >= 0) * (vecw[1] <= 1)
    if np.sometrue(tis):
        tis =  vecw[1] < 1
    return txs, tys, tzs, tis

def inv_transform(xs, ys, zs, M):
    iM = linalg.inv(M)
    vec = vec_pad_ones(xs, ys, zs)
    vecr = np.dot(iM, vec)
    try:
        vecr = vecr/vecr[3]
    except OverflowError:
        pass
    return vecr[0], vecr[1], vecr[2]

def vec_pad_ones(xs, ys, zs):
    try:
        try:
            vec = np.array([xs,ys,zs,np.ones(xs.shape)])
        except (AttributeError,TypeError):
            vec = np.array([xs,ys,zs,np.ones((len(xs)))])
    except TypeError:
        vec = np.array([xs,ys,zs,1])
    return vec

def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    """
    vec = vec_pad_ones(xs, ys, zs)
    return proj_transform_vec(vec, M)

def proj_transform_clip(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs,tys,tzs,tis
    """
    vec = vec_pad_ones(xs, ys, zs)
    return proj_transform_vec_clip(vec, M)
transform = proj_transform

def proj_points(points, M):
    return zip(*proj_trans_points(points, M))

def proj_trans_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)

def proj_trans_clip_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform_clip(xs, ys, zs, M)

def test_proj_draw_axes(M, s=1):
    import pylab
    xs, ys, zs = [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, s]
    txs, tys, tzs = proj_transform(xs, ys, zs, M)
    o, ax, ay, az = (txs[0], tys[0]), (txs[1], tys[1]), \
            (txs[2], tys[2]), (txs[3], tys[3])
    lines = [(o, ax), (o, ay), (o, az)]

    ax = pylab.gca()
    linec = LineCollection(lines)
    ax.add_collection(linec)
    for x, y, t in zip(txs, tys, ['o', 'x', 'y', 'z']):
        pylab.text(x, y, t)

def test_proj_make_M(E=None):
    # eye point
    E = E or np.array([1, -1, 2]) * 1000
    #E = np.array([20,10,20])
    R = np.array([1, 1, 1]) * 100
    V = np.array([0, 0, 1])
    viewM = view_transformation(E, R, V)
    perspM = persp_transformation(100, -100)
    M = np.dot(perspM, viewM)
    return M

def test_proj():
    import pylab
    M = test_proj_make_M()

    ts = ['%d' % i for i in [0,1,2,3,0,4,5,6,7,4]]
    xs, ys, zs = [0,1,1,0,0, 0,1,1,0,0], [0,0,1,1,0, 0,0,1,1,0], \
            [0,0,0,0,0, 1,1,1,1,1]
    xs, ys, zs = [np.array(v)*300 for v in (xs, ys, zs)]
    #
    test_proj_draw_axes(M, s=400)
    txs, tys, tzs = proj_transform(xs, ys, zs, M)
    ixs, iys, izs = inv_transform(txs, tys, tzs, M)

    pylab.scatter(txs, tys, c=tzs)
    pylab.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        pylab.text(x, y, t)

    pylab.xlim(-0.2, 0.2)
    pylab.ylim(-0.2, 0.2)

    pylab.show()

def rot_x(V, alpha):
    cosa, sina = np.cos(alpha), np.sin(alpha)
    M1 = np.array([[1,0,0,0],
                   [0,cosa,-sina,0],
                   [0,sina,cosa,0],
                   [0,0,0,0]])

    return np.dot(M1, V)

def test_rot():
    V = [1,0,0,1]
    print rot_x(V, np.pi/6)
    V = [0,1,0,1]
    print rot_x(V, np.pi/6)


if __name__ == "__main__":
    test_proj()
