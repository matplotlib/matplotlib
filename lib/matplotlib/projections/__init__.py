from geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from polar import PolarAxes
from matplotlib import axes

class ProjectionRegistry(object):
    """
    Manages the set of projections available to the system.
    """
    def __init__(self):
        self._all_projection_types = {}

    def register(self, *projections):
        """
        Register a new set of projection(s).
        """
        for projection in projections:
            name = projection.name
            self._all_projection_types[name] = projection

    def get_projection_class(self, name):
        """
        Get a projection class from its *name*.
        """
        return self._all_projection_types[name]

    def get_projection_names(self):
        """
        Get a list of the names of all projections currently
        registered.
        """
        names = self._all_projection_types.keys()
        names.sort()
        return names
projection_registry = ProjectionRegistry()

projection_registry.register(
    axes.Axes,
    PolarAxes,
    AitoffAxes,
    HammerAxes,
    LambertAxes,
    MollweideAxes)


def register_projection(cls):
    projection_registry.register(cls)

def get_projection_class(projection=None):
    """
    Get a projection class from its name.

    If *projection* is None, a standard rectilinear projection is
    returned.
    """
    if projection is None:
        projection = 'rectilinear'

    try:
        return projection_registry.get_projection_class(projection)
    except KeyError:
        raise ValueError("Unknown projection '%s'" % projection)

def projection_factory(projection, figure, rect, **kwargs):
    """
    Get a new projection instance.

    *projection* is a projection name.

    *figure* is a figure to add the axes to.

    *rect* is a :class:`~matplotlib.transforms.Bbox` object specifying
    the location of the axes within the figure.

    Any other kwargs are passed along to the specific projection
    constructor being used.
    """

    return get_projection_class(projection)(figure, rect, **kwargs)

def get_projection_names():
    """
    Get a list of acceptable projection names.
    """
    return projection_registry.get_projection_names()
