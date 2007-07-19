import numpy as npy
import enthought.traits.api as traits


class C(traits.HasTraits):
    x = traits.Float(0.0)
    y = traits.Float(1.0)

class MyC(C):
    xy = traits.Float(0.0)

    def __init__(self, c):
        self.x = c.x
        self.y = c.y

        c.sync_trait('x', self)
        c.sync_trait('y', self)

    def _x_changed(self, old, new):
        self.xy = self.x * self.y

    def _y_changed(self, old, new):
        self.xy = self.x * self.y


# C objects are created at top level
c = C()
c.x = 1
c.y = 1





class Backend:
    
    def register_c(self, c):
        # only gets C objects after creation
        self.myc = MyC(c)

backend = Backend()

backend.register_c(c)

c.x = 4

print backend.myc.xy


        
