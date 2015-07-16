Removed threading related classes from cbook
````````````````````````````````````````````
The classes ``Scheduler``, ``Timeout``, and ``Idle`` were in cbook, but
are not used internally.  They appear to be a prototype for the idle event
system which was not working and has recently been pulled out.
