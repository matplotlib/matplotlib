"""
==========
Textcursor
==========

"""
from matplotlib.widgets import TextCursor
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, facecolor='#FFFFCC')

#A linearly growing x vector.
x = np.linspace(-5, 5, 1000)
#A non biunique function. dataaxis='y' will cause trouble.
y=(x**2)

lin = ax.plot(x, y)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 25)

#A minimum call
#Set useblit=True on most backends for enhanced performance and pass the ax parameter
#to the Cursor base class.
#cursor = TextCursor(line=lin[0], ax=ax, useblit=True)

#A more advanced call. Properties for text and lines are passed.
#See the color if you are confused which parameter is passed where.
#The dataaxis parameter is still the default.
cursor = TextCursor(line=lin[0], numberformat="{0:.2f}\n{1:.2f}", dataaxis='x', offset=[10, 10], textprops={'color':'blue', 'fontweight':'bold'}, ax=ax, useblit=True, color='red', linewidth=2)

#A call demonstrating problems with the dataaxis=y parameter.
#The text now looks up the matching x value for the current cursor y position instead of vice versa.
#Hover you cursor to y=4. There are two x values producing this y value: -2 and 2.
#The function is only unique, but not biunique.
#Only one value is shown in the text.
#cursor = TextCursor(line=lin[0], numberformat="{0:.2f}\n{1:.2f}", dataaxis='y', offset=[10, 10], textprops={'color':'blue', 'fontweight':'bold'}, ax=ax, useblit=True, color='red', linewidth=2)

#In the gallery picture, we cannot see the cursor.
#The cursor is displayed after the first mouse move event. There is no mouse move when
#using automatic scripts for generating the documentation.
#The onmove event needs to occur after the plt.show() command, because this one creates a fresh
#and clean canvas. Without the cursor or its text. And after the plt.show(), we cannot execute
#more instructions. The following code draws the figure (without showing it) and calls the mouse
#event manually. The resulting figure is saved to a file. But it cannot be shown, because this
#first clears the canvas and therefore removes the cursor.
#from matplotlib.backend_bases import MouseEvent
#location = np.array([0, 10])
#location = ax.transData.transform(location)
#event = MouseEvent(name='motion_notify_event', button=None, key=None, x=location[0], y=location[1], canvas=ax.figure.canvas)
#ax.figure.canvas.draw()
#cursor.onmove(event)
#fig.savefig('text_cursor.png')

plt.show()
