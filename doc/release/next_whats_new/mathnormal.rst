Mathtext distinguishes *italic* and *normal* font
-------------------------------------------------

Matplotlib's lightweight TeX expression parser (``usetex=False``) now distinguishes between *italic* and *normal* math fonts to closer replicate the behaviour of LaTeX.
The normal math font is selected by default in math environment (unless the rcParam ``mathtext.default`` is overwritten) but can be explicitly set with the new ``\mathnormal`` command. Italic font is selected with ``\mathit``.
The main difference is that *italic* produces italic digits, whereas *normal* produces upright digits. Previously, it was not possible to typeset italic digits.
Note that ``normal`` now corresponds to what used to be ``it``, whereas ``it`` now renders all characters italic.
**Important**: In case the default mathematics font is overwritten by setting ``mathtext.default: it`` in ``matplotlibrc``, it must be either commented out or changed to ``mathtext.default: normal`` to preserve its behaviour. Otherwise, all alphanumeric characters, including digits, are rendered italic. 

One difference to traditional LaTeX is that LaTeX further distinguishes between *normal* (``\mathnormal``) and *default math*, where the default uses roman digits and normal uses oldstyle digits. This distinction is no longer present with modern LaTeX engines and unicode-math nor in Matplotlib.
