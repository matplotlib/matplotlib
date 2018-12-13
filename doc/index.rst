:orphan:

.. title:: Matplotlib: Python plotting

Matplotlib is a Python 2D plotting library which produces publication quality
figures in a variety of hardcopy formats and interactive environments across
platforms.  Matplotlib can be used in Python scripts, the Python and IPython_
shells, the Jupyter_ notebook, web application servers, and four graphical user
interface toolkits.

.. _IPython: http://ipython.org
.. _Jupyter: http://jupyter.org

.. raw:: html

   <div class="responsive_screenshots">
      <a href="tutorials/introductory/sample_plots.html">
         <div class="responsive_subfig">
         <img align="middle" src="_images/sphx_glr_membrane_thumb.png"
          border="0" alt="screenshots"/>
         </div>
         <div class="responsive_subfig">
         <img align="middle" src="_images/sphx_glr_histogram_thumb.png"
          border="0" alt="screenshots"/>
         </div>
         <div class="responsive_subfig">
         <img align="middle" src="_images/sphx_glr_contour_thumb.png"
          border="0" alt="screenshots"/>
         </div>
         <div class="responsive_subfig">
         <img align="middle" src="_images/sphx_glr_3D_thumb.png"
          border="0" alt="screenshots"/>
         </div>
      </a>
   </div>
   <span class="clear_screenshots"></span>

Matplotlib tries to make easy things easy and hard things possible.  You
can generate plots, histograms, power spectra, bar charts, errorcharts,
scatterplots, etc., with just a few lines of code.  For examples, see the
:doc:`sample plots <tutorials/introductory/sample_plots>` and :doc:`thumbnail
gallery <gallery/index>`.

For simple plotting the `pyplot` module provides a MATLAB-like interface,
particularly when combined with IPython.  For the power user, you have full
control of line styles, font properties, axes properties, etc, via an object
oriented interface or via a set of functions familiar to MATLAB users.

Installation
------------

Visit the :doc:`Matplotlib installation instructions <users/installing>`.

Documentation
-------------

This is the documentation for Matplotlib version |version|.

To get started, read the :doc:`User's Guide <users/index>`.

.. raw:: html

   <p id="other_versions"></p>

   <script>
   function getSnippet(id, url) {
      var req = false;
      // For Safari, Firefox, and other non-MS browsers
      if (window.XMLHttpRequest) {
         try {
            req = new XMLHttpRequest();
         } catch (e) {
            req = false;
         }
      } else if (window.ActiveXObject) {
         // For Internet Explorer on Windows
         try {
            req = new ActiveXObject("Msxml2.XMLHTTP");
         } catch (e) {
            try {
            req = new ActiveXObject("Microsoft.XMLHTTP");
            } catch (e) {
            req = false;
            }
         }
      }
      var element = document.getElementById(id);
      if (req) {
         // Synchronous request, wait till we have it all
         req.open('GET', url, false);
         req.send(null);
         if (req.status == 200) {
            element.innerHTML = req.responseText;
         } else {
            element.innerHTML = "<mark>Could not find Snippet to insert at " + url + "</mark>"
         }
      }
   }
   getSnippet('other_versions', '/versions.html');
   </script>

Trying to learn how to do a particular kind of plot?  Check out the
:doc:`examples gallery <gallery/index>` or the :doc:`list of plotting commands
<api/pyplot_summary>`.

Other learning resources
~~~~~~~~~~~~~~~~~~~~~~~~

There are many :doc:`external learning resources <resources/index>` available
including printed material, videos and tutorials.

Need help?
~~~~~~~~~~

Matplotlib is a welcoming, inclusive project, and we try to follow the `Python
Software Foundation Code of Conduct <coc_>`_ in everything we do.

.. _coc: http://www.python.org/psf/codeofconduct/

Check the :doc:`FAQ <faq/index>` and the :doc:`API <api/index>` docs.

For help, join the `gitter channel`_ and the matplotlib-users_,
matplotlib-devel_, and matplotlib-announce_ mailing lists, or check out the
Matplotlib tag on stackoverflow_.  The `search <search.html>`_ tool searches
all of the documentation, including full text search of over 350 complete
examples which exercise almost every corner of Matplotlib.

.. _gitter channel: https://gitter.im/matplotlib/matplotlib
.. _matplotlib-users: https://mail.python.org/mailman/listinfo/matplotlib-users
.. _matplotlib-devel: https://mail.python.org/mailman/listinfo/matplotlib-devel
.. _matplotlib-announce: https://mail.python.org/mailman/listinfo/matplotlib-announce
.. _stackoverflow: http://stackoverflow.com/questions/tagged/matplotlib

You can file bugs, patches and feature requests on the `issue tracker`_, but it
is a good idea to ping us on the mailing list too.

To keep up to date with what's going on in Matplotlib, see the :doc:`what's
new <users/whats_new>` page or browse the `source code`_.  Anything that could
require changes to your existing code is logged in the :doc:`API changes
<api/api_changes>` file.

Toolkits
========

Matplotlib ships with several add-on :doc:`toolkits <api/toolkits/index>`,
including 3d plotting with `mplot3d`, axes helpers in `axes_grid1` and axis
helpers in `axisartist`.

Third party packages
====================

A large number of :doc:`third party packages <thirdpartypackages/index>`
extend and build on Matplotlib functionality, including several higher-level
plotting interfaces (seaborn_, holoviews_, ggplot_, ...), and two projection
and mapping toolkits (basemap_ and cartopy_).

.. _seaborn: https://seaborn.github.io/
.. _holoviews: http://holoviews.org
.. _ggplot: http://ggplot.yhathq.com
.. _basemap: http://matplotlib.org/basemap
.. _cartopy: http://scitools.org.uk/cartopy/docs/latest

Citing Matplotlib
=================

Matplotlib is the brainchild of John Hunter (1968-2012), who, along with its
many contributors, have put an immeasurable amount of time and effort into
producing a piece of software utilized by thousands of scientists worldwide.

If Matplotlib contributes to a project that leads to a scientific publication,
please acknowledge this work by citing the project. A :doc:`ready-made citation
entry <citing>` is available.

Open source
===========

.. raw:: html

   <a href="https://www.numfocus.org/">
   <img src="_static/numfocus_badge.png"
    alt="A Fiscally Sponsored Project of NUMFocus"
    style="float:right; margin-left:20px" />
   </a>

Please consider `donating to the Matplotlib project <donating_>`_ through
the Numfocus organization or to the `John Hunter Technology Fellowship
<jdh-fellowship_>`_.

.. _donating: https://www.flipcause.com/secure/cause_pdetails/MjI1OA==
.. _jdh-fellowship: https://www.numfocus.org/programs/john-hunter-technology-fellowship/

The :doc:`Matplotlib license <users/license>` is based on the `Python Software
Foundation (PSF) license <psf-license_>`_.

.. _psf-license: http://www.python.org/psf/license

There is an active developer community and a long list of people who have made
significant :doc:`contributions <users/credits>`.

Matplotlib is hosted on `Github <source code_>`_.  `Issues <issue tracker_>`_
and `Pull requests`_ are tracked at Github too.

.. _source code: https://github.com/matplotlib/matplotlib
.. _issue tracker: https://github.com/matplotlib/matplotlib/issues
.. _pull requests: https://github.com/matplotlib/matplotlib/pulls
