API changes for ``constrained_layout``
----------------------------------------

The new constrained_layout functionality has some minor (largely backwards-
compatible) API changes.  See
:ref:`sphx_glr_tutorials_intermediate_constrainedlayout_guide.py` for
more details on this functionality.

This requires a new dependency on kiwisolver_.

_https://github.com/nucleic/kiwi

kwarg ``fig`` deprectated in `.GridSpec.get_subplot_params`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``figure`` instead of ``fig``, which is now deprecated. 
