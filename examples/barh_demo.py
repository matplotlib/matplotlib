#!/usr/bin/env python
# make a horizontal bar chart

from pylab import *
x = 3+10*rand(5)    # the bar lengths
y = arange(5)+.5    # the bar centers on the y axis

figure(1)
barh(x,y)
yticks(y, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
xlabel('Perfomance')
title('How fast do you want to go today?')
grid(True)

figure(2)
barh(x,y, xerr=rand(5))
yticks(y, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
xlabel('Perfomance')

show()
