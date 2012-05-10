
from enthought.traits.api import HasTraits, Str, Instance

#from enthought.traits.tests.other import Other
#from other import Other
class Simple(HasTraits):
    
    name = Str
    other = Instance('enthought.traits.tests.other.Other')
    #other = Instance(Other)
    