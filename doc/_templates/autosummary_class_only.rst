{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}


{% if objtype in ['class'] %}

.. auto{{ objtype }}:: {{ objname }}
    :no-members:

{% endif %}
