No annotation coordinates wrap
``````````````````````````````

In #2351 for 1.4.0 the behavior of ['axes points', 'axes pixel',
'figure points', 'figure pixel'] as coordinates was change to
no longer wrap for negative values.  In 1.4.3 this change was
reverted for 'axes points' and 'axes pixel' and in addition caused
'axes fraction' to wrap.  For 1.5 the behavior has been reverted to
as it was in 1.4.0-1.4.2, no wrapping for any type of coordinate.
