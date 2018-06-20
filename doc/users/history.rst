History
=======

.. note::

   The following introductory text was written in 2008 by John D. Hunter
   (1968-2012), the original author of Matplotlib.

Matplotlib is a library for making 2D plots of arrays in `Python
<https://www.python.org>`_.  Although it has its origins in emulating
the MATLAB graphics commands, it is
independent of MATLAB, and can be used in a Pythonic, object oriented
way.  Although Matplotlib is written primarily in pure Python, it
makes heavy use of `NumPy <http://www.numpy.org>`_ and other extension
code to provide good performance even for large arrays.

Matplotlib is designed with the philosophy that you should be able to
create simple plots with just a few commands, or just one!  If you
want to see a histogram of your data, you shouldn't need to
instantiate objects, call methods, set properties, and so on; it
should just work.

For years, I used to use MATLAB exclusively for data analysis and
visualization.  MATLAB excels at making nice looking plots easy.  When
I began working with EEG data, I found that I needed to write
applications to interact with my data, and developed an EEG analysis
application in MATLAB.  As the application grew in complexity,
interacting with databases, http servers, manipulating complex data
structures, I began to strain against the limitations of MATLAB as a
programming language, and decided to start over in Python.  Python
more than makes up for all of MATLAB's deficiencies as a programming
language, but I was having difficulty finding a 2D plotting package
(for 3D `VTK <http://www.vtk.org/>`_ more than exceeds all of my
needs).

When I went searching for a Python plotting package, I had several
requirements:

* Plots should look great - publication quality.  One important
  requirement for me is that the text looks good (antialiased, etc.)

* Postscript output for inclusion with TeX documents

* Embeddable in a graphical user interface for application
  development

* Code should be easy enough that I can understand it and extend
  it

* Making plots should be easy

Finding no package that suited me just right, I did what any
self-respecting Python programmer would do: rolled up my sleeves and
dived in.  Not having any real experience with computer graphics, I
decided to emulate MATLAB's plotting capabilities because that is
something MATLAB does very well.  This had the added advantage that
many people have a lot of MATLAB experience, and thus they can
quickly get up to steam plotting in python.  From a developer's
perspective, having a fixed user interface (the pylab interface) has
been very useful, because the guts of the code base can be redesigned
without affecting user code.

The Matplotlib code is conceptually divided into three parts: the
*pylab interface* is the set of functions provided by
:mod:`matplotlib.pylab` which allow the user to create plots with code
quite similar to MATLAB figure generating code
(:doc:`/tutorials/introductory/pyplot`).  The *Matplotlib frontend* or *Matplotlib
API* is the set of classes that do the heavy lifting, creating and
managing figures, text, lines, plots and so on
(:doc:`/tutorials/intermediate/artists`).  This is an abstract interface that knows
nothing about output.  The *backends* are device-dependent drawing
devices, aka renderers, that transform the frontend representation to
hardcopy or a display device (:ref:`what-is-a-backend`).  Example
backends: PS creates `PostScript®
<http://www.adobe.com/products/postscript/>`_ hardcopy, SVG
creates `Scalable Vector Graphics <https://www.w3.org/Graphics/SVG/>`_
hardcopy, Agg creates PNG output using the high quality `Anti-Grain
Geometry <http://antigrain.com/>`_
library that ships with Matplotlib, GTK embeds Matplotlib in a
`Gtk+ <https://www.gtk.org/>`_
application, GTKAgg uses the Anti-Grain renderer to create a figure
and embed it in a Gtk+ application, and so on for `PDF
<https://acrobat.adobe.com/us/en/why-adobe/about-adobe-pdf.html>`_, `WxWidgets
<https://www.wxpython.org/>`_, `Tkinter
<https://docs.python.org/library/tkinter.html>`_, etc.

Matplotlib is used by many people in many different contexts.  Some
people want to automatically generate PostScript files to send
to a printer or publishers.  Others deploy Matplotlib on a web
application server to generate PNG output for inclusion in
dynamically-generated web pages.  Some use Matplotlib interactively
from the Python shell in Tkinter on Windows. My primary use is to
embed Matplotlib in a Gtk+ EEG application that runs on Windows, Linux
and Macintosh OS X.

----

Matplotlib's original logo (2003 -- 2008).

..
   The original logo was added in fc8c215.

.. plot::

   from matplotlib import cbook, pyplot as plt, style
   import numpy as np

   style.use("classic")

   datafile = cbook.get_sample_data('membrane.dat', asfileobj=False)

   # convert data to mV
   x = 1000 * 0.1 * np.fromstring(open(datafile, 'rb').read(), np.float32)
   # 0.0005 is the sample interval
   t = 0.0005 * np.arange(len(x))
   plt.figure(1, figsize=(7, 1), dpi=100)
   ax = plt.subplot(111, facecolor='y')
   plt.plot(t, x)
   plt.text(0.5, 0.5, 'matplotlib', color='r',
            fontsize=40, fontname=['Courier', 'DejaVu Sans Mono'],
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            )
   plt.axis([1, 1.72, -60, 10])
   plt.gca().set_xticklabels([])
   plt.gca().set_yticklabels([])
