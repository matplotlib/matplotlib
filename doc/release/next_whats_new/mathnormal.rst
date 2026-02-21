Mathtext distinguishes *italic* and *normal* font
-------------------------------------------------

Matplotlib's lightweight TeX expression parser (``usetex=False``) now distinguishes between *italic* and *normal* math fonts to closer replicate the behaviour of LaTeX.
Italic font is selected with ``\mathit``,  whereas the normal math font is selected by default in math environment but can be explicitly set with the new ``\mathnormal`` command.
The main difference is that *italic* produces italic digits, whereas *normal* produces upright digits. Previously, it was not possible to typeset italic digits.

One difference to traditional LaTeX is that LaTeX further distinguishes between *normal* (``\mathnormal``) and *default math*, where the default uses roman digits and normal uses oldstyle digits. This distinction is no longer present with modern LaTeX engines and unicode-math nor in Matplotlib.
