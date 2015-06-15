import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.subplot(1,1,1)
plt.annotate('line1\nline2',xy=(0.5,0.5),xycoords='axes fraction',fontsize='xx-large',
    path_effects=[pe.withStroke(linewidth=1,foreground='r')])
plt.plot([1,.4,.5,0.2],color='blue',linewidth=5)
plt.text(2.7,0.8,'More testing',fontsize=20,color='green')

plt.show()
