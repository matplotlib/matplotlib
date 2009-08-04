    """
    Grab mpl data from the ~/.matplotlib/mpl_data cache if it exists, else
    fetch it from svn and cache it
    """
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
fname = cbook.get_mpl_data('lena.png', asfileobj=False)

print 'fname', fname
im = plt.imread(fname)
plt.imshow(im)
plt.show()
