:orphan:

.. title:: Matplotlib: Python plotting

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


Matplotlib is a Python 2D plotting library which produces publication quality
figures in a variety of hardcopy formats and interactive environments across
different platforms.

Install Matplotlib
------------------
Matplotlib can be installed on the Anaconda Python distribution, or by itself
using ``pip``. For more information see the
:doc:`installation instructions <users/installing>`.

Learn Matplotlib
----------------
Explore the functionality available in Matplotlib in the
:doc:`examples <gallery/index>`,
:doc:`tutorials <tutorials/index>`, and
:doc:`User's guide <users/index>`,

Get help
--------

Other versions
--------------
This is the documentation for Matplotlib version |version|.

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


Get involved
------------
Matplotlib is hosted on `GitHub <source code_>`_, where anyone is welcome to
contribute.

.. _source code: https://github.com/matplotlib/matplotlib

Support Matplotlib
------------------

.. raw:: html

   <a href="https://www.numfocus.org/">
   <img src="_static/numfocus_badge.png"
    alt="A Fiscally Sponsored Project of NUMFocus"
    style="float:right; margin-left:20px" />
   </a>


Matplotlib is a Sponsored Project of NumFOCUS, a 501(c)(3) nonprofit
charity in the United States. NumFOCUS provides Matplotlib with
fiscal, legal, and administrative support to help ensure the health
and sustainability of the project. Visit `numfocus.org <nf>`_ for more
information.

Donations to Matplotlib are managed by NumFOCUS. For donors in the
United States, your gift is tax-deductible to the extent provided by
law. As with any donation, you should consult with your tax adviser
about your particular tax situation.

Please consider `donating to the Matplotlib project <donating_>`_ through
the Numfocus organization or to the `John Hunter Technology Fellowship
<jdh-fellowship_>`_.

.. _donating: https://numfocus.salsalabs.org/donate-to-matplotlib/index.html
.. _jdh-fellowship: https://www.numfocus.org/programs/john-hunter-technology-fellowship/
.. _nf: https://numfocus.org
