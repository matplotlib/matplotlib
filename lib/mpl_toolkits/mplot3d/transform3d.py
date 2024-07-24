import numpy as np
import matplotlib.transforms as mtransforms


# These transforms break the assumption that the last row is [0, 0, 0, 1], and is
# therefore not affine. However, this is required to preserve the order that
# transforms are performed
class NonAffine3D(mtransforms.Affine3D):
    pass


class WorldTransform(mtransforms.Affine3D):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, pb_aspect=None):
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        if pb_aspect is not None:
            ax, ay, az = pb_aspect
            dx /= ax
            dy /= ay
            dz /= az
        mtx = np.array([
            [1/dx, 0,    0,    -xmin/dx],
            [0,    1/dy, 0,    -ymin/dy],
            [0,    0,    1/dz, -zmin/dz],
            [0,    0,    0,    1]
        ])
        super().__init__(matrix=mtx)


class PerspectiveTransform(NonAffine3D):
    def __init__(self, zfront, zback, focal_length):
        e = focal_length
        a = 1
        b = (zfront + zback) / (zfront - zback)
        c = -2 * (zfront * zback) / (zfront - zback)
        mtx = np.array([[e,   0,  0, 0],
                         [0, e/a,  0, 0],
                         [0,   0,  b, c],
                         [0,   0, -1, 0]])
        super().__init__(matrix=mtx)


class OrthographicTransform(NonAffine3D):
    def __init__(self, zfront, zback):
        a = -(zfront + zback)
        b = -(zfront - zback)
        mtx = np.array([[2, 0,  0, 0],
                         [0, 2,  0, 0],
                         [0, 0, -2, 0],
                         [0, 0,  a, b]])
        super().__init__(matrix=mtx)


class ViewTransform(mtransforms.Affine3D):
    def __init__(self, u, v, w, E):
        """
        Return the view transformation matrix.

        Parameters
        ----------
        u : 3-element numpy array
            Unit vector pointing towards the right of the screen.
        v : 3-element numpy array
            Unit vector pointing towards the top of the screen.
        w : 3-element numpy array
            Unit vector pointing out of the screen.
        E : 3-element numpy array
            The coordinates of the eye/camera.
        """
        self._u = u
        self._v = v
        self._w = w

        Mr = np.eye(4)
        Mt = np.eye(4)
        Mr[:3, :3] = [u, v, w]
        Mt[:3, -1] = -E
        mtx = np.dot(Mr, Mt)
        super().__init__(matrix=mtx)
