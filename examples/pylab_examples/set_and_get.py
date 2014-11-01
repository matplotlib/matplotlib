"""

MATLAB and pylab allow you to use setp and get to set and get
object properties, as well as to do introspection on the object

set
    To set the linestyle of a line to be dashed, you can do

      >>> line, = plot([1,2,3])
      >>> setp(line, linestyle='--')

    If you want to know the valid types of arguments, you can provide the
    name of the property you want to set without a value

      >>> setp(line, 'linestyle')
          linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]

    If you want to see all the properties that can be set, and their
    possible values, you can do

        >>> setp(line)

    set operates on a single instance or a list of instances.  If you are
    in query mode introspecting the possible values, only the first
    instance in the sequence is used.  When actually setting values, all
    the instances will be set.  e.g., suppose you have a list of two lines,
    the following will make both lines thicker and red

        >>> x = arange(0,1.0,0.01)
        >>> y1 = sin(2*pi*x)
        >>> y2 = sin(4*pi*x)
        >>> lines = plot(x, y1, x, y2)
        >>> setp(lines, linewidth=2, color='r')


get:

    get returns the value of a given attribute.  You can use get to query
    the value of a single attribute

        >>> getp(line, 'linewidth')
            0.5

    or all the attribute/value pairs

    >>> getp(line)
        aa = True
        alpha = 1.0
        antialiased = True
        c = b
        clip_on = True
        color = b
        ... long listing skipped ...

Aliases:

  To reduce keystrokes in interactive mode, a number of properties
  have short aliases, e.g., 'lw' for 'linewidth' and 'mec' for
  'markeredgecolor'.  When calling set or get in introspection mode,
  these properties will be listed as 'fullname or aliasname', as in




"""

from __future__ import print_function
from pylab import *


x = arange(0,1.0,0.01)
y1 = sin(2*pi*x)
y2 = sin(4*pi*x)
lines = plot(x, y1, x, y2)
l1, l2 = lines
setp(lines, linestyle='--')       # set both to dashed
setp(l1, linewidth=2, color='r')  # line1 is thick and red
setp(l2, linewidth=1, color='g')  # line2 is thicker and green


print ('Line setters')
setp(l1)
print ('Line getters')
getp(l1)

print ('Rectangle setters')
setp(gca().patch)
print ('Rectangle getters')
getp(gca().patch)

t = title('Hi mom')
print ('Text setters')
setp(t)
print ('Text getters')
getp(t)

show()
