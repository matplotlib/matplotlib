from enthought.traits.api import *

def Alias(name):
    return Property(lambda obj: getattr(obj, name),
                    lambda obj, val: setattr(obj, name, val))

class Path(HasTraits):
    strokecolor = Color()


class Line(Path):
   color = Alias('strokecolor')

   def __init__(self, x, color='red'):
       self.x = x
       self.color = color
       
line = Line(1)
print line.strokecolor
