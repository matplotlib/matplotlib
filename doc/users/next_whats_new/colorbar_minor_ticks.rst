Add ``minorticks_on()/off()`` methods for colorbar
--------------------------------------------------

A new method :meth:`.Colobar.minorticks_on` is
introduced to correctly display minor ticks on the colorbar. This method
doesn't allow the minor ticks to extend into the regions beyond vmin and vmax
when the extend `kwarg` (used while creating the colorbar) is set to 'both',
'max' or 'min'.
A complementary method :meth:`.Colobar.minorticks_off`
is introduced to remove the minor ticks on the colorbar.
