'''
Before fixing the bug, doing the following will give you following error:
>>> a.add('123',plt.figure().add_subplot(111))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/pymodules/python2.7/matplotlib/figure.py", line 120, in add
    Stack.remove(self, (key, a_existing))
  File "/usr/lib/pymodules/python2.7/matplotlib/cbook.py", line 1343, in remove
    raise ValueError('Unknown element o')
ValueError: Unknown element o
'''
from matplotlib import figure
import matplotlib.pyplot as plt
a = figure.AxesStack()
a.add('123', plt.figure().add_subplot(111))
print(a._elements)
a.add('123', plt.figure().add_subplot(111))
