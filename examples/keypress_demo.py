
from pylab import *

def press(event):
    print 'press', event.key
    if event.key=='g':
        grid()
        draw()
    
connect('key_press_event', press)

title('press g to toggle grid')
plot(rand(12), rand(12), 'go')
show()
