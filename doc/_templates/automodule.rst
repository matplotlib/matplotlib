{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}

{% block functions %}
{% if functions %}

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

Exceptions
----------

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}