********************
``matplotlib.hatch``
********************

.. plot::
   :include-source: false
   :alt: Example image showing hatching patterns with level 1 density.
   
   import matplotlib.pyplot as plt

   from matplotlib.patches import Rectangle

   fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))
   
   hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
   
   def hatches_plot(ax, h):
      ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=h))
      ax.text(1, -0.5, f"' {h} '", size=15, ha="center")
      ax.axis('equal')
      ax.axis('off')

   for ax, h in zip(axs.flat, hatches):
      hatches_plot(ax, h)


For examples using the hatch api refer to: :ref:`sphx_glr_gallery_shapes_and_collections_hatch_style_reference.py`


.. automodule:: matplotlib.hatch
   :members:
   :undoc-members:
   :show-inheritance: