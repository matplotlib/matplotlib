import numpy
from enthought.traits.api import HasTraits, Array, Delegate, Trait

Affine = Array('d', (3,3),
               value=numpy.array([[1,0,0], [0,1,0], [0,0,1]], numpy.float_))

class C(HasTraits):
    affine1 = Affine
    affine2 = Affine    
    affine = Affine

    def _affine1_changed(self, old, new):
        self.affine = numpy.dot(new, self.affine2)

    def _affine2_changed(self, old, new):
        self.affine = numpy.dot(self.affine1, new)


class D(HasTraits):
    affine = Delegate('c')
    c = Trait(C)

    
c = C()
d = D()
d.affine = c.affine


print 'before c', type(c.affine), c.affine
print 'before d', type(d.affine), d.affine

c.affine1 = numpy.random.rand(3,3)
print 'after c', type(c.affine), c.affine
print 'after d', type(d.affine), d.affine


