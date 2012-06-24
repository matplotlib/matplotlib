from pylab import *


subplot(211)
imshow(rand(100,100), cmap=cm.BuPu_r)
subplot(212)
imshow(rand(100,100), cmap=cm.BuPu_r)

subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = axes([0.85, 0.1, 0.075, 0.8])
colorbar(cax=cax)
show()
