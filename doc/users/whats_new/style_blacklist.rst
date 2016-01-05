Style parameter blacklist
-------------------------

In order to prevent unexpected consequences from using a style, style
files are no longer able to set parameters that affect things
unrelated to style.  These parameters include::

  'interactive', 'backend', 'backend.qt4', 'webagg.port',
  'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
  'toolbar', 'timezone', 'datapath', 'figure.max_open_warning',
  'savefig.directory', 'tk.window_focus', 'hardcopy.docstring'
