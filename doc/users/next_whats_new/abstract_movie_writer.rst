Abstract base class for movie writers
-------------------------------------

The new :class:`~matplotlib.animation.AbstractMovieWriter` class defines
the API required by a class that is to be used as the `writer` in the
`save` method of the :class:`~matplotlib.animation.Animation` class.
The existing :class:`~matplotlib.animation.MovieWriter` class now derives
from the new abstract base class.
