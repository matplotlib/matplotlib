from matplotlib.matlab import *

t = arange(0.0, 3.0, 0.05)
s = sin(2*pi*t)
styles = ('-', '--', ':', '.', 'o', '^', 'v', '<', '>', 's', '+')
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')


axisNum = 0
for row in range(5):
    for col in range(4):
        s = sin(2*pi*t)
        axisNum += 1
        subplot(5,4,axisNum)
        style = styles[axisNum % len(styles) ]
        color = colors[axisNum % len(colors) ]
        plot(t,s, style + color)
        # turn off the ticklabels if not first row or first col
        if not gca().is_first_col():
            set(gca(), 'yticklabels', [])
        if not gca().is_last_row():
            set(gca(), 'xticklabels', [])
        
show()
