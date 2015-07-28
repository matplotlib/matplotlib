Styles
------

Several new styles have been added, including many styles from the Seaborn project.
Additionally, in order to prep for the upcoming 2.0 style-change release, a 'classic' and 'default' style has been added.
For this release, the 'default' and 'classic' styles are identical.
By using them now in your scripts, you can help ensure a smooth transition during future upgrades of matplotlib, so that you can upgrade to the snazzy new defaults when you are ready! ::

    import matplotlib.style
    matplotlib.style.use('classic')

The 'default' style will give you matplotlib's latest plotting styles::

    matplotlib.style.use('default')

