"""
Illustrate some helper functions for shading regions where a logical
mask is True

See :meth:`matplotlib.collections.PolyCollection.fill_between_where`
and :meth:`matplotlib.collections.BrokenBarHCollection.span_where`
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections


t = np.arange(0.0, 2, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = 1.2*np.sin(4*np.pi*t)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('using fill_between_where')
ax.plot(t, s1, t, s2)
ax.axhline(0, color='black', lw=2)

collection = collections.PolyCollection.fill_between_where(
	   t, s1, s2, where=s1>=s2, color='green', alpha=0.5)
ax.add_collection(collection)

collection = collections.PolyCollection.fill_between_where(
	   t, s1, s2, where=s1<=s2, color='red', alpha=0.5)
ax.add_collection(collection)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('using span_where')
ax.plot(t, s1, '-')
ax.axhline(0, color='black', lw=2)

collection = collections.BrokenBarHCollection.span_where(
	   t, ymin=0, ymax=1, where=s1>0, facecolor='green', alpha=0.5)
ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
	   t, ymin=-1, ymax=0, where=s1<0, facecolor='red', alpha=0.5)
ax.add_collection(collection)



plt.show()





