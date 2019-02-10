from .. import axes, cbook
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes


class ProjectionRegistry:
    """
    Manages the set of projections available to the system.
    """
    def __init__(self):
        self._all_projection_types = {}

    def register(self, *projections):
        """
        Register a new set of projections.
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
        Get a list of the names of all projections currently registered.
        """
        return sorted(self._all_projection_types)


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

    If *projection* is None, a standard rectilinear projection is returned.
    """
    if projection is None:
        projection = 'rectilinear'

    try:
        return projection_registry.get_projection_class(projection)
    except KeyError:
        raise ValueError("Unknown projection %r" % projection)


@cbook.deprecated("3.1")
def process_projection_requirements(figure, *args, **kwargs):
    return figure._process_projection_requirements(*args, **kwargs)


def get_projection_names():
    """
    Get a list of acceptable projection names.
    """
    return projection_registry.get_projection_names()
