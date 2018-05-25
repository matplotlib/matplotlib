.. include:: ../../examples/README.txt

.. include:: ../../examples/lines_bars_and_markers/README.txt

.. exhibit::
   :srcdir: ../../examples

   lines_bars_and_markers/*.py

.. include:: ../../examples/images_contours_and_fields/README.txt

.. exhibit::
   :srcdir: ../../examples

   images_contours_and_fields/*.py

.. include:: ../../examples/subplots_axes_and_figures/README.txt

.. exhibit::
   :srcdir: ../../examples

   subplots_axes_and_figures/*.py

.. include:: ../../examples/statistics/README.txt

.. exhibit::
   :srcdir: ../../examples

   statistics/*.py

.. include:: ../../examples/pie_and_polar_charts/README.txt

.. exhibit::
   :srcdir: ../../examples

   pie_features.py
   pie_demo2.py
   pie_and_polar_charts/*.py

.. include:: ../../examples/text_labels_and_annotations/README.txt

.. exhibit::
   :srcdir: ../../examples

   text_labels_and_annotations/*.py

.. include:: ../../examples/pyplots/README.txt

.. exhibit::
   :srcdir: ../../examples

   pyplots/*.py

.. include:: ../../examples/color/README.txt

.. exhibit::
   :srcdir: ../../examples

   color/color_demo.py
   color/*.py

.. include:: ../../examples/shapes_and_collections/README.txt

.. exhibit::
   :srcdir: ../../examples

   shapes_and_collections/*.py

.. include:: ../../examples/style_sheets/README.txt

.. exhibit::
   :srcdir: ../../examples

   style_sheets/*.py

.. include:: ../../examples/axes_grid1/README.txt

.. exhibit::
   :srcdir: ../../examples

   axes_grid1/*.py

.. include:: ../../examples/axisartist/README.txt

.. exhibit::
   :srcdir: ../../examples

   axisartist/*.py

.. include:: ../../examples/showcase/README.txt

.. exhibit::
   :srcdir: ../../examples

   showcase/*.py

.. jinja:: default

   {% for subdir in examples %}
   .. include:: ../../examples/{{ subdir }}/README.txt

   .. exhibit::
      :srcdir: ../../examples

      {{ subdir }}/*.py
   {% endfor %}
