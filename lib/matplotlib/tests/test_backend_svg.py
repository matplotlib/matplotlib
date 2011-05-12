import matplotlib.pyplot as plt
import numpy as np
import cStringIO as StringIO
import xml.parsers.expat
from matplotlib.testing.decorators import knownfailureif, cleanup

@cleanup
def test_visibility():
    # This is SF 2856495. See
    # https://sourceforge.net/tracker/?func=detail&aid=2856495&group_id=80706&atid=560720
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    x = np.linspace(0,4*np.pi,50)
    y = np.sin(x)
    yerr = np.ones_like(y)

    a,b,c=ax.errorbar(x,y,yerr=yerr,fmt='ko')
    for artist in b:
        artist.set_visible(False)

    fd = StringIO.StringIO()
    fig.savefig(fd, format='svg')

    fd.seek(0)
    buf = fd.read()
    fd.close()

    parser = xml.parsers.expat.ParserCreate()
    parser.Parse(buf) # this will raise ExpatError if the svg is invalid
