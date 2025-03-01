Specifying font feature tags
----------------------------

OpenType fonts may support feature tags that specify alternate glyph shapes or
substitutions to be made optionally. The text API now supports setting a list of feature
tags to be used with the associated font. Feature tags can be set/get with:

- `matplotlib.text.Text.set_fontfeatures` / `matplotlib.text.Text.get_fontfeatures`
- Any API that creates a `.Text` object by passing the *fontfeatures* argument (e.g.,
  ``plt.xlabel(..., fontfeatures=...)``)

Font feature strings are eventually passed to HarfBuzz, and so all `string formats
supported by hb_feature_from_string()
<https://harfbuzz.github.io/harfbuzz-hb-common.html#hb-feature-from-string>`__ are
supported. Note though that subranges are not explicitly supported and behaviour may
change in the future.

For example, the default font ``DejaVu Sans`` enables Standard Ligatures (the ``'liga'``
tag) by default, and also provides optional Discretionary Ligatures (the ``dlig`` tag.)
These may be toggled with ``+`` or ``-``.

.. plot::
    :include-source:

    fig = plt.figure(figsize=(7, 3))

    fig.text(0.5, 0.85, 'Ligatures', fontsize=40, horizontalalignment='center')

    # Default has Standard Ligatures (liga).
    fig.text(0, 0.6, 'Default: fi ffi fl st', fontsize=40)

    # Disable Standard Ligatures with -liga.
    fig.text(0, 0.35, 'Disabled: fi ffi fl st', fontsize=40,
             fontfeatures=['-liga'])

    # Enable Discretionary Ligatures with dlig.
    fig.text(0, 0.1, 'Discretionary: fi ffi fl st', fontsize=40,
             fontfeatures=['dlig'])

Available font feature tags may be found at
https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
