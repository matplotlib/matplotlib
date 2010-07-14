matplotlib for MacOS X 10.3.9 or later and Python 2.5 and Python 2.6

matplotlib is a python 2D plotting library which produces publication
quality figures in a variety of hardcopy formats and interactive
environments across platforms. matplotlib can be used in python
scripts, the python and ipython shell (ala matlab or mathematica), web
application servers, and various graphical user interface toolkits.

Home page: <http://matplotlib.sourceforge.net/>

Before running matplotlib, you must install numpy.  Binary installers
for all these packages are available here:

  <http://pythonmac.org/packages/py25-fat/index.html>.

*** Back Ends ***

You may use TkAgg or WXAgg back ends; Qt and GTK support is not
provided in this package. By default this matplotlib uses TkAgg
because Tcl/Tk is included with MacOS X.

If you wish to use WXAgg then:
* Install wxPython from:
  <http://pythonmac.org/packages/py25-fat/index.html>.
* Configure a matplotlibrc file, as described below.

For TkAgg you may use Apple's built-in Tcl/Tk or install your own 8.4.x

*** Configuring a matplotlibrc file ***

If you wish to change any matplotlib settings, create a file:
  ~/.matplotlib/matplotlibrc


that contains at least the following information. The values shown are
the defaults in the internal matplotlibrc file; change them as you see
fit:

# the default backend; one of GTK GTKAgg GTKCairo FltkAgg QtAgg TkAgg WXAgg
#     Agg Cairo GD GDK Paint PS PDF SVG Template
backend      : TkAgg
interactive  : False    # see http://matplotlib.sourceforge.net/interactive.html

See also
<http://matplotlib.sourceforge.net/users/customizing.html>

