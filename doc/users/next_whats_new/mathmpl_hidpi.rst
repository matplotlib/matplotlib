More configuration of ``mathmpl:`` sphinx extension
---------------------------------------------------

The `matplotlib.sphinxext.mathmpl` sphinx extension supports two new
configuration options that may be specified in your ``conf.py``:

- ``mathmpl_fontsize`` (float), which sets the font size of the math text in
  points;
- ``mathmpl_srcset`` (list of str), which provides a list of sizes to support
  `responsive resolution images
  <https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images>`__
  The list should contain additional x-descriptors (``'1.5x'``, ``'2x'``, etc.)
  to generate (1x is the default and always included.)
