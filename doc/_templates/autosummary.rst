{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}


{% if objtype in ['class'] %}

.. auto{{ objtype }}:: {{ objname }}
    :show-inheritance:
    :special-members: __call__

{% else %}
.. auto{{ objtype }}:: {{ objname }}

{% endif %}

{% if objtype in ['class', 'method', 'function'] %}
{% if objname in ['AxesGrid', 'Scalable', 'HostAxes', 'FloatingAxes',
'ParasiteAxesAuxTrans', 'ParasiteAxes'] %}

.. Filter out the above aliases to other classes, as sphinx gallery
   creates no example file for those (sphinx-gallery/sphinx-gallery#365)

{% else %}

.. minigallery:: {{module}}.{{objname}}
   :add-heading:

{% endif %}
{% endif %}
