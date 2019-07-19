rcParams for default axes title location and color
--------------------------------------------------

Two new rcParams have been added: ``axes.titlelocation`` denotes the default axes title
alignment, and ``axes.titlecolor`` the default axes title color.

Valid values for ``axes.titlelocation`` are: left, center, and right.
Valid values for ``axes.titlecolor`` are: auto or a color. Setting it to auto
will fall back to previous behaviour, which is using the color in ``text.color``.
