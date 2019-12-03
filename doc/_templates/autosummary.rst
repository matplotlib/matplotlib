{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}


{% if objtype in ['class'] %}
{% if module in ['matplotlib.image'] %}
.. auto{{ objtype }}:: {{ objname }}
    :show-inheritance:
    :inherited-members:
{% else %}
.. auto{{ objtype }}:: {{ objname }}
    :show-inheritance:
{% endif %}
{% endif %}

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
