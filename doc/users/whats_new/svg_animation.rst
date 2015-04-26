Added SVG backend for animations
--------------------------------

SVG is available as a backend for animations.  Select ``svg`` as the writer: ::

    animation.writers['svg']

SVG animation makes use of the ``animate`` and ``set`` XML elements as
specified at WC3_.  The animation starts on page load and restarts whenever
the ``svg`` element is clicked.

.. _WC3: http://www.w3.org/TR/SVG/animate.html

