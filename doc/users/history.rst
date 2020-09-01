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
makes heavy use of `NumPy <https://numpy.org>`_ and other extension
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
:mod:`pylab` which allow the user to create plots with code
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

Matplotlib logo (2008 - 2015).

..
   This logo was added in 325e47b.

.. plot::

   import numpy as np
   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import matplotlib.cm as cm

   mpl.rcParams['xtick.labelsize'] = 10
   mpl.rcParams['ytick.labelsize'] = 12
   mpl.rcParams['axes.edgecolor'] = 'gray'


   axalpha = 0.05
   figcolor = 'white'
   dpi = 80
   fig = plt.figure(figsize=(6, 1.1), dpi=dpi)
   fig.patch.set_edgecolor(figcolor)
   fig.patch.set_facecolor(figcolor)


   def add_math_background():
       ax = fig.add_axes([0., 0., 1., 1.])

       text = []
       text.append(
           (r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = "
            r"U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2}"
            r"\int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 "
            r"\left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - "
            r"\alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} "
            r"}{U^{0\beta}_{\rho_1 \sigma_2}}\right]$", (0.7, 0.2), 20))
       text.append((r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} "
                    r"= -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$",
                    (0.35, 0.9), 20))
       text.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$",
                    (0.15, 0.3), 25))
       text.append((r"$F_G = G\frac{m_1m_2}{r^2}$",
                    (0.85, 0.7), 30))
       for eq, (x, y), size in text:
            ax.text(x, y, eq, ha='center', va='center', color="#11557c",
                   alpha=0.25, transform=ax.transAxes, fontsize=size)
       ax.set_axis_off()
       return ax


   def add_matplotlib_text(ax):
       ax.text(0.95, 0.5, 'matplotlib', color='#11557c', fontsize=65,
               ha='right', va='center', alpha=1.0, transform=ax.transAxes)


   def add_polar_bar():
       ax = fig.add_axes([0.025, 0.075, 0.2, 0.85], projection='polar')

       ax.patch.set_alpha(axalpha)
       ax.set_axisbelow(True)
       N = 7
       arc = 2. * np.pi
       theta = np.arange(0.0, arc, arc/N)
       radii = 10 * np.array([0.2, 0.6, 0.8, 0.7, 0.4, 0.5, 0.8])
       width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
       bars = ax.bar(theta, radii, width=width, bottom=0.0)
       for r, bar in zip(radii, bars):
           bar.set_facecolor(cm.jet(r/10.))
           bar.set_alpha(0.6)

       ax.tick_params(labelbottom=False, labeltop=False,
                      labelleft=False, labelright=False)

       ax.grid(lw=0.8, alpha=0.9, ls='-', color='0.5')

       ax.set_yticks(np.arange(1, 9, 2))
       ax.set_rmax(9)


   main_axes = add_math_background()
   add_polar_bar()
   add_matplotlib_text(main_axes)
