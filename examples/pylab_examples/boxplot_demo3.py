import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

np.random.seed(2)
inc = 0.1
e1 = np.random.uniform(0,1, size=(500,))
e2 = np.random.uniform(0,1, size=(500,))
e3 = np.random.uniform(0,1 + inc, size=(500,))
e4 = np.random.uniform(0,1 + 2*inc, size=(500,))

treatments = [e1,e2,e3,e4]

fig = plt.figure()
ax = fig.add_subplot(111)
pos = np.array(range(len(treatments)))+1
bp = ax.boxplot( treatments, sym='k+', patch_artist=True,
                 positions=pos, notch=1, bootstrap=5000 )
text_transform= mtransforms.blended_transform_factory(ax.transData,
                                                     ax.transAxes)
ax.set_xlabel('treatment')
ax.set_ylabel('response')
ax.set_ylim(-0.2, 1.4)
plt.setp(bp['whiskers'], color='k',  linestyle='-' )
plt.setp(bp['fliers'], markersize=3.0)
fig.subplots_adjust(right=0.99,top=0.99)
plt.show()
