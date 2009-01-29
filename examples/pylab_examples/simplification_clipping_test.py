from pylab import *
import numpy as np

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s, linewidth=1.0)

ylim((-0.20, -0.28))

title('Should see four lines extending from bottom to top')

figure()

x = np.array([1.0,2.0,3.0,2.0e5])
y = np.arange(len(x))
plot(x,y)
xlim(xmin=2,xmax=6)
title("Should be monotonically increasing")

figure()

x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
y = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
plot(x, y)
xlim(xmin=-0.6, xmax=0.6)
ylim(ymin=-0.6, ymax=0.6)
title("Diamond shape, with segments visible in all four corners")

figure()

np.random.seed(0)
x = np.random.uniform(size=(5000,)) * 50

rcParams['path.simplify'] = True
p1 = plot(x,solid_joinstyle='round',linewidth=2.0)

path = p1[0].get_path()
transform = p1[0].get_transform()
path = transform.transform_path(path)
simplified = list(path.iter_segments(simplify=(800, 600)))

title("Original length: %d, simplified length: %d" % (len(path.vertices), len(simplified)))

figure()

x = np.sin(np.linspace(0, np.pi * 2.0, 1000)) + np.random.uniform(size=(1000,)) * 0.01

rcParams['path.simplify'] = True
p1 = plot(x,solid_joinstyle='round',linewidth=2.0)

path = p1[0].get_path()
transform = p1[0].get_transform()
path = transform.transform_path(path)
simplified = list(path.iter_segments(simplify=(800, 600)))

title("Original length: %d, simplified length: %d" % (len(path.vertices), len(simplified)))

show()
