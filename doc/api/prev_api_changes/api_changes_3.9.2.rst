API Changes for 3.9.2
=====================

Development
-----------

Windows wheel runtime bundling made static
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 3.7.0, the MSVC runtime DLL was bundled in wheels to enable importing Matplotlib on
systems that do not have it installed. However, this could cause inconsistencies with
other wheels that did the same, and trigger random crashes depending on import order. See
`this issue <https://github.com/matplotlib/matplotlib/issues/28551>`_ for further
details.

Since 3.9.2, wheels now bundle the MSVC runtime DLL statically to avoid such issues.
