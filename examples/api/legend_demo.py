import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0,3,.02)
b = np.arange(0,3,.02)
c = np.exp(a)
d = c[::-1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(a,c,'k--',a,d,'k:',a,c+d,'k')
leg = ax.legend(('Model length', 'Data length', 'Total message length'),
           'upper center', shadow=True)
ax.set_ylim([-1,20])
ax.grid(False)
ax.set_xlabel('Model complexity --->')
ax.set_ylabel('Message length --->')
ax.set_title('Minimum Message Length')

ax.set_yticklabels([])
ax.set_xticklabels([])

# set some legend properties.  All the code below is optional.  The
# defaults are usually sensible but if you need more control, this
# shows you how

# the matplotlib.patches.Rectangle instance surrounding the legend
frame  = leg.get_frame()  
frame.set_facecolor('0.80')    # set the frame face color to light gray

# matplotlib.text.Text instances
for t in leg.get_texts():
    t.set_fontsize('small')    # the legend text fontsize

# matplotlib.lines.Line2D instances
for l in leg.get_lines():
    l.set_linewidth(1.5)  # the legend line width
plt.show()



