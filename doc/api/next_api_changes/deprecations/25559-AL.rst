Accessing ``event.guiEvent`` after event handlers return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated: for some GUI toolkits, it is unsafe to do so.  In the
future, ``event.guiEvent`` will be set to None once the event handlers return;
you may separately stash the object at your own risk.
