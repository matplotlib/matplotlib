.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry
   :class: multicol-toc

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

Methods
^^^^^^^

    .. autosummary::
       :nosignatures:
    {% for item in methods %}
       ~{{ name }}.{{ item }}
    {% endfor %}

    {% endif %}
    {% endblock %}

Documentation
=============

.. autoclass:: {{ objname }}
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Examples
========

.. include:: {{module}}.{{name}}.examples
{% block examples %}
{% if methods %}
{% for item in methods %}
.. include:: {{module}}.{{name}}.{{item}}.examples
.. raw:: html

    <div class="clearer"></div>
{% endfor %}
{% endif %}
{% endblock %}

