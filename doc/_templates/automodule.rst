{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block classes %}
{% if classes %}

Classes
-------

.. autosummary::
   :template: autosummary.rst
   :toctree:

{% for item in classes %}{% if item not in ['zip', 'map', 'reduce'] %}
   {{ item }}{% endif %}{% endfor %}

{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}

Functions
---------

.. autosummary::
   :template: autosummary.rst
   :toctree:

{% for item in functions %}{% if item not in ['zip', 'map', 'reduce'] %}
   {{ item }}{% endif %}{% endfor %}

{% endif %}
{% endblock %}
