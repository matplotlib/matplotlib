:orphan:

.. title:: Matplotlib: Python plotting

Matplotlib: Visualization with Python
-------------------------------------

Matplotlib is a comprehensive library for creating static, animated,
and interactive visualizations in Python.

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

Matplotlib makes easy things easy and hard things possible.

.. container:: bullet-box-container

   .. container:: bullet-box

      Create

      - Develop `publication quality plots`_ with just a few lines of code
      - Use `interactive figures`_ that can zoom, pan, update...

      .. _publication quality plots: https://matplotlib.org/gallery/index.html
      .. _interactive figures: https://matplotlib.org/gallery/index.html#event-handling

   .. container:: bullet-box

      Customize

      - `Take full control`_ of line styles, font properties, axes properties...
      - `Export and embed`_ to a number of file formats and interactive environments

      .. _Take full control: https://matplotlib.org/tutorials/index.html#tutorials
      .. _Export and embed: https://matplotlib.org/api/index_backend_api.html

   .. container:: bullet-box

      Extend

      - Explore tailored functionality provided by
        :doc:`third party packages <thirdpartypackages/index>`
      - Learn more about Matplotlib through the many
        :doc:`external learning resources <resources/index>`

Documentation
~~~~~~~~~~~~~

To get started, read the :doc:`User's Guide <users/index>`.

Trying to learn how to do a particular kind of plot?  Check out the
:doc:`examples gallery <gallery/index>` or the :doc:`list of plotting commands
<api/pyplot_summary>`.

Join our community!
~~~~~~~~~~~~~~~~~~~

Matplotlib is a welcoming, inclusive project, and we follow the `Python
Software Foundation Code of Conduct <coc_>`_ in everything we do.

.. _coc: https://www.python.org/psf/conduct/


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
        <p>Check out the Matplotlib tag on <a href="https://stackoverflow.com/questions/tagged/matplotlib">stackoverflow</a>.</p>
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
        <ul>
        <li>Tweet us at <a href="https://twitter.com/matplotlib">@matplotlib</a>!</li>
        <li>See cool plots on <a href="https://www.instagram.com/matplotart/">@matplotart</a> Instagram!</li>
        <li>Check out our <a href="https://matplotlib.org/matplotblog/">Blog</a>!</li>
        </ul>
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
including 3D plotting with `.mplot3d`, axes helpers in `.axes_grid1` and axis
helpers in `.axisartist`.

Third party packages
====================

A large number of :doc:`third party packages <thirdpartypackages/index>`
extend and build on Matplotlib functionality, including several higher-level
plotting interfaces (seaborn_, HoloViews_, ggplot_, ...), and a projection
and mapping toolkit (Cartopy_).

.. _seaborn: https://seaborn.pydata.org
.. _HoloViews: https://holoviews.org
.. _ggplot: http://ggplot.yhathq.com
.. _Cartopy: https://scitools.org.uk/cartopy/docs/latest

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

   <a href="https://numfocus.org/">
   <img src="_static/numfocus_badge.png"
    alt="A Fiscally Sponsored Project of NUMFocus"
    style="float:right; margin-left:20px" />
   </a>


Matplotlib is a Sponsored Project of NumFOCUS, a 501(c)(3) nonprofit
charity in the United States. NumFOCUS provides Matplotlib with
fiscal, legal, and administrative support to help ensure the health
and sustainability of the project. Visit `numfocus.org <nf_>`_ for more
information.

Donations to Matplotlib are managed by NumFOCUS. For donors in the
United States, your gift is tax-deductible to the extent provided by
law. As with any donation, you should consult with your tax adviser
about your particular tax situation.

Please consider `donating to the Matplotlib project <donating_>`_ through
the NumFOCUS organization or to the `John Hunter Technology Fellowship
<jdh-fellowship_>`_.

.. _donating: https://numfocus.org/donate-to-matplotlib
.. _jdh-fellowship: https://numfocus.org/programs/john-hunter-technology-fellowship/
.. _nf: https://numfocus.org

The :doc:`Matplotlib license <users/license>` is based on the `Python Software
Foundation (PSF) license <psf-license_>`_.

.. _psf-license: https://www.python.org/psf/license

There is an active developer community and a long list of people who have made
significant :doc:`contributions <users/credits>`.
