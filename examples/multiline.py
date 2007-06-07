#!/usr/bin/env python
from pylab import *

plot(arange(10))
xlabel('this is a xlabel\n(with newlines!)')
ylabel('this is vertical\ntest', multialignment='center')
#ylabel('this is another!')
text(2, 7,'this is\nyet another test',
     rotation=45,
     horizontalalignment = 'center',
     verticalalignment   = 'top',
     multialignment      = 'center')
#savefig('multiline')
grid(True)
show()
