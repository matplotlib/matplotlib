from matplotlib.pyplot import *

subplot(211)
plot([1,2,3], label="test1")
plot([3,2,1], label="test2")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)

subplot(223)
plot([1,2,3], label="test1")
plot([3,2,1], label="test2")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


show()
