"""
=======
Textbox
=======

Allowing text input with the Textbox widget.

You can use the Textbox widget to let users interactively provide any text 
that needs to be displayed, including formulas. You can use a submit button to 
create plots with the given input.

Note to not get confused with 
:doc:`/tutorials/text/annotations` and 
:doc:`/gallery/recipes/placing_text_boxes`, both of which are static elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(-2.0, 2.0, 0.001)
s = t ** 2
initial_text = "t ** 2"
l, = plt.plot(t, s, lw=2)


def submit(text):
    # user can enter a new math expression that uses "t" as its independent 
    # variable. The plot refreshes on entering. Example: try "t ** 3"
    ydata = eval(text)  
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
text_box.on_submit(submit)

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

from matplotlib.widgets import TextBox
