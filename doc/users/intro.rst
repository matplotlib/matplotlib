Introduction
============

matplotlib is a library for making 2D plots of arrays in `Python
<http://www.python.org>`_.  Although it has its origins in emulating
the MATLAB |reg| [*]_ graphics commands, it is
independent of MATLAB, and can be used in a Pythonic, object oriented
way.  Although matplotlib is written primarily in pure Python, it
makes heavy use of `NumPy <http://www.numpy.org>`_ and other extension
code to provide good performance even for large arrays.

.. |reg| unicode:: 0xAE
   :ltrim:

matplotlib is designed with the philosophy that you should be able to
create simple plots with just a few commands, or just one!  If you
want to see a histogram of your data, you shouldn't need to
instantiate objects, call methods, set properties, and so on; it
should just work.

For years, I used to use MATLAB exclusively for data analysis and
visualization.  MATLAB excels at making nice looking plots easy.  When
I began working with EEG data, I found that I needed to write
applications to interact with my data, and developed and EEG analysis
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

The matplotlib code is conceptually divided into three parts: the
*pylab interface* is the set of functions provided by
:mod:`matplotlib.pylab` which allow the user to create plots with code
quite similar to MATLAB figure generating code
(:ref:`pyplot-tutorial`).  The *matplotlib frontend* or *matplotlib
API* is the set of classes that do the heavy lifting, creating and
managing figures, text, lines, plots and so on
(:ref:`artist-tutorial`).  This is an abstract interface that knows
nothing about output.  The *backends* are device dependent drawing
devices, aka renderers, that transform the frontend representation to
hardcopy or a display device (:ref:`what-is-a-backend`).  Example
backends: PS creates `PostScript®
<http://www.adobe.com/products/postscript/>`_ hardcopy, SVG
creates `Scalable Vector Graphics <http://www.w3.org/Graphics/SVG/>`_
hardcopy, Agg creates PNG output using the high quality `Anti-Grain
Geometry <http://www.antigrain.com>`_ library that ships with
matplotlib, GTK embeds matplotlib in a `Gtk+ <http://www.gtk.org/>`_
application, GTKAgg uses the Anti-Grain renderer to create a figure
and embed it a Gtk+ application, and so on for `PDF
<http://www.adobe.com/products/acrobat/adobepdf.html>`_, `WxWidgets
<http://www.wxpython.org/>`_, `Tkinter
<http://docs.python.org/lib/module-Tkinter.html>`_ etc.

matplotlib is used by many people in many different contexts.  Some
people want to automatically generate PostScript files to send
to a printer or publishers.  Others deploy matplotlib on a web
application server to generate PNG output for inclusion in
dynamically-generated web pages.  Some use matplotlib interactively
from the Python shell in Tkinter on Windows™. My primary use is to
embed matplotlib in a Gtk+ EEG application that runs on Windows, Linux
and Macintosh OS X.

.. [*] MATLAB is a registered trademark of The MathWorks, Inc.


