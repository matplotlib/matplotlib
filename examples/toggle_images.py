""" toggle between two images by pressing "b"

The basic idea is to load two images (they can be different shapes) and plot them to the same axes with hold "on".  Then, toggle the visible property of them using keypress event handling

As usual, we'll define some random images for demo.  Real data is much more exciting
"""

from pylab import *
# two images x1 is visible, x2 is not
x1 = rand(100,100)
x2 = rand(150, 175)

im1 = imshow(x1, interpolation='nearest')
im2 = imshow(x2, interpolation='nearest', hold=True)
im2.set_visible(False)

def toggle_images(event):
    'toggle the visible state of the two images'
    if event.key != 'b': return
    b1 = im1.get_visible()
    b2 = im2.get_visible()
    im1.set_visible(not b1)
    im2.set_visible(not b2)
    draw()

connect('key_press_event', toggle_images)
#savefig('toggle_images')
show()
