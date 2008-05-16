These examples use the matplotlib api rather than the pylab/pyplot
procedural state machine.  For robust, production level scripts, or
for applications or web application servers, we recommend you use the
matplotlib API directly as it gives you the maximum control over your
figures, axes and plottng commands.  There are a few documentation
resources for the API

  - the matplotlib artist tutorial :
    http://matplotlib.sourceforge.net/pycon/artist_api_tut.pdf

  - the "leftwich tutorial" -
    http://matplotlib.sourceforge.net/leftwich_tut.txt

 The example agg_oo.py is the simplest example of using the Agg
 backend which is readily ported to other output formats.  This
 example is a good starting point if your are a web application
 developer.  Many of the other examples in this directory use
 matplotlib.pyplot just to create the figure and show calls, and use
 the API for everything else.  This is a good solution for production
 quality scripts.  For full fledged GUI applications, see the
 user_interfaces examples.
