{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :show-inheritance:

{% if '__init__' in methods %}
{% set caught_result = methods.remove('__init__') %}
{% endif %}

{% block methods %}
{% if methods %}

.. rubric:: Methods

.. autosummary::
   :template: autosummary.rst
   :toctree:
   :nosignatures:
{% for item in methods %}
   ~{{ name }}.{{ item }}
{% endfor %}

{% endif %}
{% endblock %}