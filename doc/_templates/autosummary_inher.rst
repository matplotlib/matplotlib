{{ fullname | escape | underline}}


.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    :inherited-members:
    :show-inheritance:

{% if objtype in ['class', 'method', 'function'] %}
{% if objname in ['AxesGrid', 'Scalable', 'HostAxes', 'FloatingAxes',
                      'ParasiteAxesAuxTrans', 'ParasiteAxes'] %}
.. Filter out the above aliases to other classes, as sphinx gallery
   creates no example file for those (sphinx-gallery/sphinx-gallery#365)

{% else %}
.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>

{% endif %}
{% endif %}