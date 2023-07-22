``Text.get_rotation_mode`` return value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``None`` as ``rotation_mode`` to `.Text` (the default value) or passing it to
`.Text.set_rotation_mode` will make `.Text.get_rotation_mode` return ``"default"``
instead of ``None``. The behaviour otherwise is the same.
