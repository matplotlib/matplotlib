Improve Label Properties Settings in ``RadioButtons``
-----------------------------------------------------

Since styling was introduced to `RadioButtons` in version 3.7.0, the
``label_props`` argument of `~.RadioButtons.__init__` and the
`~.RadioButtons.set_label_props` class method, both required a single element
list per every property in the given dictionary. Now these dictionary values
can be a scalar, which will be used to all radio button labels, or a list. In
case of a list, every value in the list will be used per label, and the list's
length should fit the amount of labels. A list with an improper amount of items
raises an error message.
