Event objects emitted for ``axes_leave_event``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``axes_leave_event`` now emits a synthetic `.LocationEvent`, instead of reusing
the last event object associated with a ``motion_notify_event``.
