from geo import AitoffAxes, HammerAxes, LambertAxes
from polar import PolarAxes
from matplotlib import axes

class ProjectionRegistry(object):
    def __init__(self):
        self._all_projection_types = {}

    def register(self, *projections):
        for projection in projections:
            name = projection.name
            self._all_projection_types[name] = projection

    def get_projection_class(self, name):
        return self._all_projection_types[name]

    def get_projection_names(self):
        names = self._all_projection_types.keys()
        names.sort()
        return names
projection_registry = ProjectionRegistry()

projection_registry.register(
    axes.Axes,
    PolarAxes,
    AitoffAxes,
    HammerAxes,
    LambertAxes)


def register_projection(cls):
    projection_registry.register(cls)

def get_projection_class(projection):
    if projection is None:
        projection = 'rectilinear'

    try:
        return projection_registry.get_projection_class(projection)
    except KeyError:
        raise ValueError("Unknown projection '%s'" % projection)

def projection_factory(projection, figure, rect, **kwargs):
    return get_projection_class(projection)(figure, rect, **kwargs)

def get_projection_names():
    return projection_registry.get_projection_names()
