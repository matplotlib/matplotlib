from .. import axes
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


def process_projection_requirements(
        figure, *args, polar=False, projection=None, **kwargs):
    """
    Handle the args/kwargs to add_axes/add_subplot/gca, returning::

        (axes_proj_class, proj_class_kwargs, proj_stack_key)

    which can be used for new axes initialization/identification.
    """
    if polar:
        if projection is not None and projection != 'polar':
            raise ValueError(
                "polar=True, yet projection=%r. "
                "Only one of these arguments should be supplied." %
                projection)
        projection = 'polar'

    if isinstance(projection, str) or projection is None:
        projection_class = get_projection_class(projection)
    elif hasattr(projection, '_as_mpl_axes'):
        projection_class, extra_kwargs = projection._as_mpl_axes()
        kwargs.update(**extra_kwargs)
    else:
        raise TypeError('projection must be a string, None or implement a '
                        '_as_mpl_axes method. Got %r' % projection)

    # Make the key without projection kwargs, this is used as a unique
    # lookup for axes instances
    key = figure._make_key(*args, **kwargs)

    return projection_class, kwargs, key


def get_projection_names():
    """
    Get a list of acceptable projection names.
    """
    return projection_registry.get_projection_names()
