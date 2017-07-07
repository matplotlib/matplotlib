
{{ fullname | escape | underline }}


.. currentmodule:: {{ fullname }}

{% block functions %}
{% if functions %}

Functions
---------

.. autosummary:: 
   :template: autosummary.rst
   :toctree:
{% for item in functions %}{% if item not in ['plotting', 'colormaps'] %}
   {{ item }}{% endif %}{% endfor %}
{% endif %}
{% endblock %}
