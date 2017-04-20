{{ fullname | escape | underline}}


.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{% if objtype in ['class', 'method', 'function'] %}

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>

{% endif %}