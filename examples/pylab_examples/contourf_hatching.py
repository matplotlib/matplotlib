import matplotlib.pyplot as plt
import numpy

x = numpy.linspace(-3, 5, 150).reshape(1, -1)
y = numpy.linspace(-3, 5, 120).reshape(-1, 1)
z = numpy.cos(x) + numpy.sin(y)

# we no longer need x and y to be 2 dimensional, so flatten them
x = x.flatten()
y = y.flatten()

# plot #1
# the simplest hatched plot
fig = plt.figure()
cm = plt.contourf(x, y, z, hatches=['-', '/', '\\', '//'], cmap=plt.get_cmap('gray'),
                  extend='both', alpha=0.5
                  )

plt.colorbar()


# plot #2
# a plot of hatches without color
plt.figure()
plt.contour(x, y, z, colors='black', )
plt.contourf(x, y, z, colors='none', hatches=['.', '/', '\\', None, '..', '\\\\'])


plt.show()