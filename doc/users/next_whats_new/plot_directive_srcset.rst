Plot Directive now can make responsive images with "srcset"
-----------------------------------------------------------

The plot sphinx directive (``matplotlib.sphinxext.plot_directive``, invoked in
rst as ``.. plot::``) can be configured to automatically make higher res
figures and add these to the the built html docs.  In ``conf.py``::

    extensions = [
    ...
        'matplotlib.sphinxext.plot_directive',
        'matplotlib.sphinxext.figmpl_directive',
    ...]

    plot_srcset = ['2x']

will make png files with double the resolution for hiDPI displays.  Resulting
html files will have image entries like::

    <img src="../_images/nestedpage-index-2.png" style="" srcset="../_images/nestedpage-index-2.png, ../_images/nestedpage-index-2.2x.png 2.00x" alt="" class="plot-directive "/>
