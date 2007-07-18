import numpy
from enthought.traits.api import HasTraits, Array
import mtraits


class Path(HasTraits):
    """
    The path is an object that talks to the backends, and is an
    intermediary between the high level path artists like Line and
    Polygon, and the backend renderer
    """
    strokecolor = mtraits.color('white')

p = Path()
print 'strokecolor', p.strokecolor
