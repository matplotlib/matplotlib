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

To get started, read the :doc:`User's Guide <users/index>`.

Trying to learn how to do a particular kind of plot?  Check out the
:doc:`examples gallery <gallery/index>` or the :doc:`list of plotting commands
<api/pyplot_summary>`.

Other learning resources
~~~~~~~~~~~~~~~~~~~~~~~~

There are many :doc:`external learning resources <resources/index>` available
including printed material, videos and tutorials.

Join our community!
~~~~~~~~~~~~~~~~~~~

Matplotlib is a welcoming, inclusive project, and we try to follow the `Python
Software Foundation Code of Conduct <coc_>`_ in everything we do.

.. _coc: http://www.python.org/psf/codeofconduct/


.. raw:: html

    <h3>Get help</h3>
    <div class="box">
      <div class="box-item">
        <img src="_static/fa/discourse-brands.svg" alt="Discourse">
        <p>Join our community at <a href="https://discourse.matplotlib.org">discourse.matplotlib.org</a>
        to get help, discuss contributing &amp; development, and share your work.</p>
      </div>
      <div class="box-item">
        <img src="_static/fa/question-circle-regular.svg" alt="Questions">
        <p>If you have questions, be sure to check the <a href="faq/index.html">FAQ</a>,
        the <a href="api/index.html">API</a> docs. The full text
        <a href="search.html">search</a> is a good way to discover the docs including the many examples.</p>
      </div>
      <div class="box-item">
        <img src="_static/fa/stack-overflow-brands.svg" alt="Stackoverflow">
        <p>Check out the Matplotlib tag on <a href="http://stackoverflow.com/questions/tagged/matplotlib">stackoverflow</a>.</p>
      </div>
      <div class="box-item">
        <img src="_static/fa/gitter-brands.svg" alt="Gitter">
        <p>Short questions may be posted on the <a href="https://gitter.im/matplotlib/matplotlib">gitter channel</a>.</p>
      </div>
    </div>
    <hr class='box-sep'>
    <h3>News</h3>
    <div class="box">
      <div class="box-item">
        <img src="_static/fa/plus-square-regular.svg" alt="News">
        <p>To keep up to date with what's going on in Matplotlib, see the
        <a href="users/whats_new.html">what's new</a> page or browse the
        <a href="https://github.com/matplotlib/matplotlib">source code</a>.  Anything that could
        require changes to your existing code is logged in the
        <a href="api/api_changes.html">API changes</a> file.</p>
      </div>
      <div class="box-item">
        <img src="_static/fa/hashtag-solid.svg" alt="Social media">
        <p>Tweet us at <a href="https://twitter.com/matplotlib">Twitter</a>!
        or see cool plots on <a href="https://www.instagram.com/matplotart/">Instagram</a>!</p>
      </div>
    </div>
    <hr class='box-sep'>
    <h3>Development</h3>
    <div class="box">
      <div class="box-item">
        <img src="_static/fa/github-brands.svg" alt="Github">
        <p>Matplotlib is hosted on <a href="https://github.com/matplotlib/matplotlib">GitHub</a>.</p>
        <ul>
        <li>File bugs and feature requests on the <a href="https://github.com/matplotlib/matplotlib/issues">issue tracker</a>.</li>
        <li><a href="https://github.com/matplotlib/matplotlib/pulls">Pull requests</a> are always welcome.</li>
        </ul>
        <p>It is a good idea to ping us on <a href="https://discourse.matplotlib.org">Discourse</a> as well.</p>
      </div>
      <div class="box-item">
        <img src="_static/fa/envelope-regular.svg" alt="Mailing lists">
        <p>Mailing lists</p>
        <ul>
        <li><a href="https://mail.python.org/mailman/listinfo/matplotlib-users">matplotlib-users</a> for usage questions</li>
        <li><a href="https://mail.python.org/mailman/listinfo/matplotlib-devel">matplotlib-devel</a> for development</li>
        <li><a href="https://mail.python.org/mailman/listinfo/matplotlib-announce">matplotlib-announce</a> for project announcements</li>
        </ul>
      </div>
    </div>


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
