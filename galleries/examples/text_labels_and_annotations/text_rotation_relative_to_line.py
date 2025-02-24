"""
=======================================
Text rotation angle in data coordinates
=======================================

Text objects in matplotlib are normally rotated with respect to the
screen coordinate system (i.e., 45 degrees rotation plots text along a
line that is in between horizontal and vertical no matter how the axes
are changed).  However, at times one wants to rotate text with respect
to something on the plot.  In this case, the correct angle won't be
the angle of that object in the plot coordinate system, but the angle
that that object APPEARS in the screen coordinate system.  This angle
can be determined automatically by setting the parameter
*transform_rotates_text*, as shown in the example below.
"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Plot diagonal line (45 degrees in data coordinates)
ax.plot(range(0, 8), range(0, 8))
ax.set_xlim([-10, 10])

# Plot text
ax.text(-8, 0, 'text 45° in screen coordinates', fontsize=18,
        rotation=45, rotation_mode='anchor')
ax.text(0, 0, 'text 45° in data coordinates', fontsize=18,
        rotation=45, rotation_mode='anchor',
        transform_rotates_text=True)

plt.show()
