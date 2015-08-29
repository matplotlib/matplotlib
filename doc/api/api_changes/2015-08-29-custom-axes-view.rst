New API for custom Axes view changes
````````````````````````````````````

Interactive pan and zoom were previously implemented using a Cartesian-specific
algorithm that was not necessarily applicable to custom Axes. Three new private
methods, :meth:`~matplotlib.axes._base._AxesBase._get_view`,
:meth:`~matplotlib.axes._base._AxesBase._set_view`, and
:meth:`~matplotlib.axes._base._AxesBase._set_view_from_bbox`, allow for custom
``Axes`` classes to override the pan and zoom algorithms. Implementors of
custom ``Axes`` who override these methods may provide suitable behaviour for
both pan and zoom as well as the view navigation buttons on the interactive
toolbars. 
