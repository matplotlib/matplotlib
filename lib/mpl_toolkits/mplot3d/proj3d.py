"""
Various transforms used for by the 3D code
"""



import numpy as np
from matplotlib import _api

# Transformation function to convert world coordinates to normalized device coordinates
def world_transformation(xmin, xmax, ymin, ymax, zmin, zmax, pb_aspect=None):
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # Check if plotbox aspect is specified
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    # Calculate the inverse values for scaling
    dx_inv, dy_inv, dz_inv = 1 / dx, 1 / dy, 1 / dz

    # Return the transformation matrix
    return np.array([[dx_inv, 0, 0, -xmin * dx_inv],
                     [0, dy_inv, 0, -ymin * dy_inv],
                     [0, 0, dz_inv, -zmin * dz_inv],
                     [0, 0, 0, 1]])

# Function to calculate a rotation matrix about a given vector
def rotation_about_vector(v, angle):
    # Normalize the vector
    vx, vy, vz = v / np.linalg.norm(v)

    # Calculate sine, cosine, and intermediate values
    s = np.sin(angle)
    c = np.cos(angle)
    t = 2 * np.sin(angle / 2) ** 2

    t_vx = t * vx
    t_vy = t * vy
    t_vz = t * vz
    t_vxvx = t_vx * vx
    t_vyvy = t_vy * vy
    t_vzvz = t_vz * vz
    t_vxvy = t_vx * vy
    t_vxvz = t_vx * vz
    t_vyvz = t_vy * vz
    vx_s = vx * s
    vy_s = vy * s
    vz_s = vz * s

    # Fill in the rotation matrix
    c1 = t_vxvx + c
    c2 = t_vxvy - vz_s
    c3 = t_vxvz + vy_s
    c4 = t_vyvx + vz_s
    c5 = t_vyvy + c
    c6 = t_vyvz - vx_s
    c7 = t_vzvx - vy_s
    c8 = t_vzvy + vx_s
    c9 = t_vzvz + c

    R = np.array([
        [c1, c2, c3],
        [c4, c5, c6],
        [c7, c8, c9]
    ])

    return R

# Function to calculate the view transformation matrix
def view_transformation(E, R, V, roll):
    # Calculate the viewing axes
    w = (E - R)
    w = w / np.linalg.norm(w)
    u = np.cross(V, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    # Apply roll if it's not zero
    if roll != 0:
        Rroll = rotation_about_vector(w, -roll)
        u = np.dot(Rroll, u)
        v = np.dot(Rroll, v)

    # Create and return the view transformation matrix
    Mr = np.eye(4)
    Mt = np.eye(4)
    Mr[:3, :3] = np.column_stack((u, v, w))
    Mt[:3, -1] = -E
    M = np.dot(Mr, Mt)
    return M

# Function to create a perspective projection matrix
def persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront + zback) / (zfront - zback)
    c = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[e, 0, 0, 0],
                     [0, e / a, 0, 0],
                     [0, 0, b, c],
                     [0, 0, -1, 0]])

# Function to create an orthographic projection matrix
def ortho_transformation(zfront, zback):
    a = -(zfront + zback)
    b = -(zfront - zback)
    return np.array([[2, 0, 0, 0],
                     [0, 2, 0, 0],
                     [0, 0, -2, 0],
                     [0, 0, a, b]])

# Function to transform a vector using a projection matrix
def proj_transform_vec(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    return txs, tys, tzs

# Function to transform a vector using a projection matrix and perform clipping
def proj_transform_vec_clip(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w

    # Perform clipping
    tis = (0 <= vecw[0]) & (vecw[0] <= 1) & (0 <= vecw[1]) & (vecw[1] <= 1)
    if np.any(tis):
        tis = vecw[1] < 1

    return txs, tys, tzs, tis

# Function to perform an inverse transformation using the inverse projection matrix
def inv_transform(xs, ys, zs, invM):
    vec = np.column_stack((xs, ys, zs, np.ones_like(xs)))
    vecr = np.dot(invM, vec)

    # Vectorized division
    vecr = vecr / vecr[3]

    # Return the transformed coordinates
    return vecr[:, 0], vecr[:, 1], vecr[:, 2]

# Function to transform a set of points using a projection matrix
def proj_transform(xs, ys, zs, M):
    vec = np.column_stack((xs, ys, zs, np.ones_like(xs))
    vecw = np.dot(M, vec)
    w = vecw[:, 3]
    txs, tys, tzs = vecw[:, 0] / w, vecw[:, 1] / w, vecw[:, 2] / w
    return txs, tys, tzs

# Function to transform a set of points using a projection matrix and perform clipping
def proj_transform_clip(xs, ys, zs, M):
    vec = np.column_stack((xs, ys, zs, np.ones_like(xs))
    vecw = np.dot(M, vec)
    w = vecw[:, 3]
    txs, tys, tzs = vecw[:, 0] / w, vecw[:, 1] / w, vecw[:, 2] / w

    # Perform clipping
    tis = (0 <= vecw[:, 0]) & (vecw[:, 0] <= 1) & (0 <= vecw[:, 1]) & (vecw[:, 1] <= 1)
    if np.any(tis):
        tis = vecw[:, 1] < 1

    return txs, tys, tzs, tis

# Function to transform a set of points using a projection matrix and return the transformed points
def proj_points(points, M):
    vec = np.column_stack(points).T
    vecw = np.dot(M, vec)
    w = vecw[3]
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    return np.column_stack((txs, tys, tzs))

# Function to rotate a vector around the X-axis
def rot_x(V, alpha):
    cosa, sina = np.cos(alpha), np.sin(alpha)
    M1 = np.array([[1, 0, 0, 0],
                   [0, cosa, -sina, 0],
                   [0, sina, cosa, 0],
                   [0, 0, 0, 1]])

    # Apply the rotation and return the transformed vector
    return np.dot(M1, V)


