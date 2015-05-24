Remove chunk logic from backend_agg.draw_path and rcparams
``````````````````````````````````````````````````````````

Removes some old (~2008) experimental logic to chop up long paths in
to shorter chunks before drawing.  This has largely been superseded by
path simplification.

This behavior has always been off by default.
