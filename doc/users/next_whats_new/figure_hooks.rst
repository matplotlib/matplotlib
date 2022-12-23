Figure hooks
~~~~~~~~~~~~
The new :rc:`figure.hooks` provides a mechanism to register
arbitrary customizations on pyplot figures; it is a list of
"dotted.module.name:dotted.callable.name" strings specifying functions
that are called on each figure created by `.pyplot.figure`; these
functions can e.g. attach callbacks or modify the toolbar.  See
:doc:`/gallery/user_interfaces/mplcvd` for an example of toolbar customization.
