New rcParams for Axes creation
------------------------------

A number of rcParams are introduced to control parts of the Axes creation.
For a standard 2D Axes, the following are introduced:

* :rc:`axes.adjustable`, see `.Axes.set_adjustable`
* :rc:`axes.anchor`, see `.Axes.set_anchor`
* :rc:`axes.aspect`, see `.Axes.set_aspect`
* :rc:`axes.box_aspect`, see `.Axes.set_box_aspect`

There are separate parameters for 3D Axes, including an additional 3D-specific one:

* :rc:`axes3d.adjustable`, see `.Axes.set_adjustable`
* :rc:`axes3d.anchor`, see `.Axes.set_anchor`
* :rc:`axes3d.aspect`, see `.Axes3D.set_aspect`
* :rc:`axes3d.box_aspect`, see `.Axes3D.set_box_aspect`
* :rc:`axes3d.proj_type`, see `.Axes3D.set_proj_type`
