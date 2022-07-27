"""
Various transforms used for by the 3D code
"""

import numpy as np
import numpy.linalg as linalg


def _line2d_seg_dist(p1, p2, p0):
    """
    Return the distance(s) from line defined by p1 - p2 to point(s) p0.

    p0[0] = x(s)
    p0[1] = y(s)

    intersection point p = p1 + u*(p2-p1)
    and intersection point lies within segment if u is between 0 and 1.

    If p1 and p2 are identical, the distance between them and p0 is returned.
    """

    x01 = np.asarray(p0[0]) - p1[0]
    y01 = np.asarray(p0[1]) - p1[1]
    if np.all(p1[0:2] == p2[0:2]):
        return np.hypot(x01, y01)

    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    u = (x01*x21 + y01*y21) / (x21**2 + y21**2)
    u = np.clip(u, 0, 1)
    d = np.hypot(x01 - u*x21, y01 - u*y21)

    return d


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    return np.array([[1/dx, 0,    0,    -xmin/dx],
                     [0,    1/dy, 0,    -ymin/dy],
                     [0,    0,    1/dz, -zmin/dz],
                     [0,    0,    0,    1]])


def rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / np.linalg.norm(v)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 2*np.sin(angle/2)**2  # more numerically stable than t = 1-c

    R = np.array([
        [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],
        [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
        [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c]])

    return R


def view_transformation(E, R, V, roll):
    n = (E - R)
    n = n/np.linalg.norm(n)
    u = np.cross(V, n)
    u = u/np.linalg.norm(u)
    v = np.cross(n, u)  # Will be a unit vector

    # Save some computation for the default roll=0
    if roll != 0:
        # A positive rotation of the camera is a negative rotation of the world
        Rroll = rotation_about_vector(n, -roll)
        u = np.dot(Rroll, u)
        v = np.dot(Rroll, v)

    Mr = np.eye(4)
    Mt = np.eye(4)
    Mr[:3, :3] = [u, v, n]
    Mt[:3, -1] = -E

    return np.dot(Mr, Mt)


def persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront+zback)/(zfront-zback)
    c = -2*(zfront*zback)/(zfront-zback)
    proj_matrix = np.array([[e,   0,  0, 0],
                            [0, e/a,  0, 0],
                            [0,   0,  b, c],
                            [0,   0, -1, 0]])
    return proj_matrix


def ortho_transformation(zfront, zback):
    # note: w component in the resulting vector will be (zback-zfront), not 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    proj_matrix = np.array([[2, 0,  0, 0],
                            [0, 2,  0, 0],
                            [0, 0, -2, 0],
                            [0, 0,  a, b]])
    return proj_matrix


def _proj_transform_vec(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs


def _proj_transform_vec_clip(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here.
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    tis = (0 <= vecw[0]) & (vecw[0] <= 1) & (0 <= vecw[1]) & (vecw[1] <= 1)
    if np.any(tis):
        tis = vecw[1] < 1
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, M):
    iM = linalg.inv(M)
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = np.dot(iM, vec)
    try:
        vecr = vecr / vecr[3]
    except OverflowError:
        pass
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    return np.array([xs, ys, zs, np.ones_like(xs)])


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


transform = proj_transform


def proj_transform_clip(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs, tys, tzs, tis
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M)


def proj_points(points, M):
    return np.column_stack(proj_trans_points(points, M))


def proj_trans_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)


def rot_x(V, alpha):
    cosa, sina = np.cos(alpha), np.sin(alpha)
    M1 = np.array([[1, 0, 0, 0],
                   [0, cosa, -sina, 0],
                   [0, sina, cosa, 0],
                   [0, 0, 0, 1]])
    return np.dot(M1, V)
