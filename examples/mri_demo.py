from matplotlib.matlab import *

# data are 256x256 16 bit integers
dfile = 'data/s1045.ima'
im = fromstring(file(dfile, 'rb').read(), Int16).astype(Float)
im.shape = 256, 256

imshow(im, ColormapJet(256))
axis('off')
#savefig('mri_demo')
show()
