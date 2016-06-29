GTK and GDK backends deprecated
```````````````````````````````
The untested and broken GDK and GTK backends have been deprecated.
These backends allows figures to be rendered via the GDK api to
files and GTK2 figures. They are untested, known to be broken and 
use have been discouraged for some time. The `GTKAgg` and `GTKCairo` backends
provide better and more tested ways of rendering figures to GTK2 windows.

WX backend deprecated
`````````````````````
The untested WX backend has been deprecated.
This backend allows figures to be rendered via the WX api to
files and Wx figures. It is untested, and 
use have been discouraged for some time. The `WXAgg` backend
provides a better and more tested way of rendering figures to WX windows.

CocoaAgg backend removed
````````````````````````

The deprecated and not fully functional CocoaAgg backend has been removed
