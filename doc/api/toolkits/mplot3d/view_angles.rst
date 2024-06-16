.. _toolkit_mplot3d-view-angles:

*******************
mplot3d View Angles
*******************

How to define the view angle
============================

The position of the viewport "camera" in a 3D plot is defined by three angles:
*azimuth*, *elevation*, and *roll*. From the resulting position, it always
points towards the center of the plot box volume. The angle direction is a
common convention, and is shared with
`PyVista <https://docs.pyvista.org/api/core/camera.html>`_ and
`MATLAB <https://www.mathworks.com/help/matlab/ref/view.html>`_.
Note that a positive roll angle rotates the
viewing plane clockwise, so the 3d axes will appear to rotate
counter-clockwise.

.. image:: /_static/mplot3d_view_angles.png
   :align: center
   :scale: 50

Rotating the plot using the mouse will control azimuth, elevation,
as well as roll, and all three angles can be set programmatically::

    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(azim=45, elev=30, roll=15)

Rotation of the plot
====================

The *azim*, *elev*, *roll* rotation order corresponds to rotation of the scene
observed by a stationary camera. First, a left-handed rotation about the z axis is
applied (*azim*), then a right-handed rotation about the (camera) y axis (*elev*), then a
right-handed rotation about the (camera) x axis (*roll*). Here, the z, y, and x axis are fixed
axes (not the axes that rotate together with the original scene).

This can also be thought of as orbiting a camera around a fixed scene, by reversing
the order of operations. First the camera is rotated about the scene's +x axis
(*roll*), then the +y axis (*elev*), then the −z axis (*azim*).

If you would like to make the connection with quaternions (because
`Euler angles are horrible <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>`_):
the *azim*, *elev*, *roll* angles relate to the (intrinsic) rotation of the plot via:

     *q* = exp( +roll **x̂** / 2) exp( +elev **ŷ** / 2) exp( −azim **ẑ** / 2)

(with angles given in radians instead of degrees). That is, the angles are a kind of
Tait-Bryan angles: −z, +y', +x", rather than classic Euler angles.


Primary view planes
===================

To look directly at the primary view planes, the required azimuth, elevation,
and roll angles are shown in the diagram of an "unfolded" plot below. These are
further documented in the `.mplot3d.axes3d.Axes3D.view_init` API.

.. plot:: gallery/mplot3d/view_planes_3d.py
   :align: center
