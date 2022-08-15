Font fallback in Agg
--------------------

It is now possible to specify a list of fonts families and the Agg renderer
will try them in order to locate a required glyph.

.. plot::
   :caption: Demonstration of mixed English and Chinese text with font fallback.
   :alt: The phrase "There are 几个汉字 in between!" rendered in various fonts.
   :include-source: True

   import matplotlib.pyplot as plt

   text = "There are 几个汉字 in between!"

   plt.rcParams["font.size"] = 20
   fig = plt.figure(figsize=(4.75, 1.85))
   fig.text(0.05, 0.85, text, family=["WenQuanYi Zen Hei"])
   fig.text(0.05, 0.65, text, family=["Noto Sans CJK JP"])
   fig.text(0.05, 0.45, text, family=["DejaVu Sans", "Noto Sans CJK JP"])
   fig.text(0.05, 0.25, text, family=["DejaVu Sans", "WenQuanYi Zen Hei"])

   plt.show()


This currently only works with the Agg backend, but support for the vector
backends is planned for Matplotlib 3.7.
