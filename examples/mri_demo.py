from matplotlib.matlab import *

# data are 256x256 16 bit integers
dfile = 'data/s1045.ima'
im = fromstring(file(dfile).read(), Int16).astype(Float)
im.shape = 256, 256
# flip upside down
im = array([im[i] for i in arange(255,-1,-1)])

pcolor(im, shading='flat')
axis('off')
savefig('mri_demo')
show()
