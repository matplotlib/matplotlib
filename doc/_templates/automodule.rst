{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block functions %}
{% if functions %}

Classes
-------

.. autosummary:: 
   :template: autosummary.rst
   :toctree:

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}

Functions
---------

.. autosummary:: 
   :template: autosummary.rst
   :toctree:

{% for item in functions %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}

Exceptions
----------

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}