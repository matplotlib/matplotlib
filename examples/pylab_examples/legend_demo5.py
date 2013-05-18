import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

fig, (ax1, ax2) = plt.subplots(2, 1)
p1 = ax1.scatter([1],[5], c='r', marker='s', s=100)
p2 = ax1.scatter([3],[2], c='b', marker='o', s=100)

l = ax1.legend([(p1, p2)],['points'], scatterpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=0)})

ind = [1,2,3]
pos1 = [1, 3, 2]
neg1 = [2, 1, 4]
width=[0.5, 0.5, 0.5]

rpos1 = ax2.bar(ind, pos1, width=0.5, color='k', label='+1')
rneg1 = ax2.bar(ind, neg1, width=0.5, color='w', hatch='///', label='-1')

l = ax2.legend([(rpos1, rneg1)],['Test'],
               handler_map={(rpos1, rneg1): HandlerTuple(ndivide=0, pad=0.)})

plt.show()
