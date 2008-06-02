*****
HOWTO
*****

How do I use matplotlib in a web application server?
====================================================

Many users report initial problems trying to use maptlotlib in web
application servers, because by default matplotlib ships configured to
work with a graphical user interface which may require an X11
connection.  Since many barebones application servers do not have X11
enabled, you may get errors if you don't configure matplotlib for use
in these environments.  Most importantly, you need to decide what
kinds of images you want to generate (PNG, PDF, SVG) and configure the
appropriate default backend.  For 99% of users, this will be the Agg
backend, which uses the C++ `antigrain <http://antigrain.com>`_
rendering engine to make nice PNGs.  The Agg backend is also
configured to recognize requests to generate other output formats
(PDF, PS, EPS, SVG).  The easiest way to configure matplotlib to use
Agg is to call::

    # do this before importing pylab or pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

Alternatively, you can avoid pylab/pyplot altogeher, which will give
you a little more control, by calling the API directly as shown in
`agg_oo.py <http://matplotlib.sf.net/examples/api/agg_oo.py>`_ .

You can either generate hardcopy on the filesystem by calling savefig::

    # do this before importing pylab or pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    fig.savefig('test.png')

or by saving to a file handle::

    import sys
    fig.savefig(sys.stdout)


How do I use matplotlib with apache?
------------------------------------

TODO

How do I use matplotlib with dhango?
------------------------------------

TODO

How do I use matplotlib with zope?
----------------------------------

TODO
