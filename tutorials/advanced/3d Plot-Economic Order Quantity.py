from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
def get_test_data(delta=0.05):
    from matplotlib.mlab import bivariate_normal
    X=500
    Y=250
    Z=1000
    L=3.5
    D=2
    return X,Y,Z,L,D
fig=plt.figure()

ax=fig.add_subplot(111,projection='3d')
X,Y,Z=axes3d.get_test_data(0.05)
ax.plot_wireframe(X,Y,Z,rstride=10, cstride=10,color='indigo')
ax.set_xlabel('Backorder items', fontsize=10,color="orange")
ax.set_ylabel('rework rate for \ndefective items',fontsize=10,color="violet")
ax.set_zlabel('Backorder cost\n per item', fontsize=10,color="maroon", rotation = 0)
plt.title('3d Plot-Economic Order Quantity ')
plt.show()

