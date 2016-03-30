#!/usr/bin/python

# This module adds functionality to plot spherical polygons and is a result of
# the discussion of issue #5294 (Handling Spherical Polygons) on github. The
# following people contributed to this discussion:
# Nikolai Nowaczyk, Tyler Reddy, Nicolas P. Rougier, Phillip Wolfram,
# Thomas A Caswell

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class SphericalPolygon(object):
    """
    Plots a spherical polygon onto a sphere.
    """

    def __init__(self,
                 vertices,
                 tri=np.array([[0, 1, 2]]),
                 center=np.array([0, 0, 0]),
                 radius=1.,
                 subdivisions=None):
        """
        Parameters
        ----------
        vertices:
            an array of points in 3D assumed to be on a sphere
        tri:
            a np.array [[i,j,k], ...] of triples [i,j,k] of integers i,j,k,
            where i, j, k are the indices of the vertices defining the face.
            (If left empty, class will assume only 3 vertices in clockwise
            order.) See also note below.
        center:
            a point in 3D defining the center of the sphere
        radius:
            a positive double representing the radius of the sphere
        subdivisions:
            an integer defining how often the triangulation of
            the polygon is subdivided before it is plotted. (If left
            empty, the class will try to guess this value such that
            the result looks nice.)

        Note (computing triangulations)
        ------------------------------
        In case no triangulation of the polygon is available, it can be
        easily computed for instance like this:

        >>> from scipy.spatial import ConvexHull
        >>> tri = ConvexHull(vertices).simplices
        >>> sp = SphericalPolygon(vertices, tri)

        """

        self.vertices = vertices
        self.center = center
        self.radius = radius
        self.subdivisions = subdivisions
        self.tri = tri
        self._refine_triangulation()

    def _project_to_sphere(self, point):
        """
        Projects point onto the sphere defined by self.center and self.radius.

        Parameters
        ----------
        point:
            array of floats of length ndim

        Returns
        ----------
            array of floats of length ndim
        """

        length = np.linalg.norm(point - self.center)
        return (point - self.center) / length * self.radius + self.center

    def _subdivide(self):
        """
        Subdivides the triangulation once by replacing every triangle
        by four smaller triangles defined by the vertices of the original
        triangle and the midpoints of its edges.
        """

        faces_subdivided = []

        for face in self.tri:
            # create three new vertices at the midpoints of each edge
            a = self.vertices[face[0]]
            b = self.vertices[face[1]]
            c = self.vertices[face[2]]

            # add vertices to list
            k = len(self.vertices)
            self.vertices = np.insert(self.vertices, k,
                                      self._project_to_sphere(a + b), axis=0)
            self.vertices = np.insert(self.vertices, k + 1,
                                      self._project_to_sphere(b + c), axis=0)
            self.vertices = np.insert(self.vertices, k + 2,
                                      self._project_to_sphere(a + c), axis=0)

            # add new faces
            faces_subdivided.append([face[0], k, k + 2])
            faces_subdivided.append([k, face[1], k + 1])
            faces_subdivided.append([k + 1, face[2], k + 2])
            faces_subdivided.append([k, k + 1, k + 2])

        self.tri = faces_subdivided

    def _refine_triangulation(self):
        """
         Refines the triangulation of the vertices.
        """

        if self.subdivisions is None:
            while np.min(np.array([min(
                np.linalg.norm(self.vertices[face[0]] - self.vertices[face[1]]),
                np.linalg.norm(self.vertices[face[0]] - self.vertices[face[2]]),
                np.linalg.norm(self.vertices[face[1]] - self.vertices[face[2]]))
                                   for face in self.tri])) > self.radius * 0.2:
                self._subdivide()

        else:
            for n in range(0, self.subdivisions):
                self._subdivide()

    def add_to_ax(self, ax, *args, **kwargs):
        """
         Adds each triangle of the triangulation to an ax object.
        """
        for face in self.tri:
            ax.add_collection3d(
                Poly3DCollection([self.vertices[face]], *args, **kwargs))
