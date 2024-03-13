Check and Radio Button widgets support clearing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.CheckButtons` and `.RadioButtons` widgets now support clearing their
state by calling their ``.clear`` method. Note that it is not possible to have
no selected radio buttons, so the selected option at construction time is selected.
