Change in the ``draw_image`` backend API
----------------------------------------

The ``draw_image`` method implemented by backends has changed its interface.

This change is only relevant if the backend declares that it is able
to transform images by returning ``True`` from ``option_scale_image``.
See the ``draw_image`` docstring for more information.
