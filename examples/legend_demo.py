# Thanks to Charles Twardy for this example
from matplotlib.matlab import *

a = arange(0,3,.02)
b = arange(0,3,.02)
c=exp(a)
d=c.tolist()
d.reverse()
d = array(d)

ax = subplot(111)
plot(a,c,'k--',a,d,'k:',a,c+d,'k')
legend(('Model length', 'Data length', 'Total message length'),
       'upper center')
ax.set_ylim([-1,20])
ax.grid(0)
xlabel('Model complexity --->')
ylabel('Message length --->')
title('Minimum Message Length')
set(gca(), 'yticklabels', [])
set(gca(), 'xticklabels', [])

#savefig('legend_demo')
show()



