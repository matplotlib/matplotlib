{{ fullname | escape | underline}}

{% if fullname in ['mpl_toolkits.axes_grid1.colorbar'] %}
.. To prevent problems with the autosummary for the colorbar doc
   treat this separately (sphinx-doc/sphinx/issues/4874)
.. automodule:: {{ fullname }}
   :members:

{% else %}

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
{% endif %}
