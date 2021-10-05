.. _github-stats-3-3-3:

GitHub statistics for 3.3.3 (Nov 11, 2020)
==========================================

GitHub statistics for 2020/09/15 (tag: v3.3.2) - 2020/11/11

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed 14 issues and merged 46 pull requests.
The full list can be seen `on GitHub <https://github.com/matplotlib/matplotlib/milestone/58?closed=1>`__

The following 11 authors contributed 73 commits.

* Antony Lee
* David Stansby
* Elliott Sales de Andrade
* Eric Larson
* Jody Klymak
* Jouni K. Seppänen
* Ryan May
* shevawen
* Stephen Sinclair
* Thomas A Caswell
* Tim Hoffmann

GitHub issues and pull requests:

Pull Requests (46):

* :ghpull:`18936`: Backport PR #18929 on branch v3.3.x
* :ghpull:`18929`: FIX: make sure scalarmappable updates are handled correctly in 3D
* :ghpull:`18928`: Backport PR #18842 on branch v3.3.x (Add CPython 3.9 wheels.)
* :ghpull:`18842`: Add CPython 3.9 wheels.
* :ghpull:`18921`: Backport PR #18732 on branch v3.3.x (Add a ponyfill for ResizeObserver on older browsers.)
* :ghpull:`18732`: Add a ponyfill for ResizeObserver on older browsers.
* :ghpull:`18886`: Backport #18860 on branch v3.3.x
* :ghpull:`18860`: FIX: stop deprecation message colorbar
* :ghpull:`18845`: Backport PR #18839 on branch v3.3.x
* :ghpull:`18843`: Backport PR #18756 on branch v3.3.x (FIX: improve date performance regression)
* :ghpull:`18850`: Backport CI fixes to v3.3.x
* :ghpull:`18839`: MNT: make sure we do not mutate input in Text.update
* :ghpull:`18838`: Fix ax.set_xticklabels(fontproperties=fp)
* :ghpull:`18756`: FIX: improve date performance regression
* :ghpull:`18787`: Backport PR #18769 on branch v3.3.x
* :ghpull:`18786`: Backport PR #18754 on branch v3.3.x (FIX: make sure we have more than 1 tick with small log ranges)
* :ghpull:`18754`: FIX: make sure we have more than 1 tick with small log ranges
* :ghpull:`18769`: Support ``ax.grid(visible=<bool>)``.
* :ghpull:`18778`: Backport PR #18773 on branch v3.3.x (Update to latest cibuildwheel release.)
* :ghpull:`18773`: Update to latest cibuildwheel release.
* :ghpull:`18755`: Backport PR #18734 on branch v3.3.x (Fix deprecation warning in GitHub Actions.)
* :ghpull:`18734`: Fix deprecation warning in GitHub Actions.
* :ghpull:`18725`: Backport PR #18533 on branch v3.3.x
* :ghpull:`18723`: Backport PR #18584 on branch v3.3.x (Fix setting 0-timeout timer with Tornado.)
* :ghpull:`18676`: Backport PR #18670 on branch v3.3.x (MNT: make certifi actually optional)
* :ghpull:`18670`: MNT: make certifi actually optional
* :ghpull:`18665`: Backport PR #18639 on branch v3.3.x (nbagg: Don't close figures for bubbled events.)
* :ghpull:`18639`: nbagg: Don't close figures for bubbled events.
* :ghpull:`18640`: Backport PR #18636 on branch v3.3.x (BLD: certifi is not a run-time dependency)
* :ghpull:`18636`: BLD: certifi is not a run-time dependency
* :ghpull:`18629`: Backport PR #18621 on branch v3.3.x (Fix singleshot timers in wx.)
* :ghpull:`18621`: Fix singleshot timers in wx.
* :ghpull:`18607`: Backport PR #18604 on branch v3.3.x (Update test image to fix Ghostscript 9.53.)
* :ghpull:`18604`: Update test image to fix Ghostscript 9.53.
* :ghpull:`18584`: Fix setting 0-timeout timer with Tornado.
* :ghpull:`18550`: backport pr 18549
* :ghpull:`18545`: Backport PR #18540 on branch v3.3.x (Call to ExitStack.push should have been ExitStack.callback.)
* :ghpull:`18549`: FIX: unit-convert pcolorargs before interpolating
* :ghpull:`18540`: Call to ExitStack.push should have been ExitStack.callback.
* :ghpull:`18533`: Correctly remove support for \stackrel.
* :ghpull:`18509`: Backport PR #18505 on branch v3.3.x (Fix depth shading when edge/facecolor is none.)
* :ghpull:`18505`: Fix depth shading when edge/facecolor is none.
* :ghpull:`18504`: Backport PR #18500 on branch v3.3.x (BUG: Fix all-masked imshow)
* :ghpull:`18500`: BUG: Fix all-masked imshow
* :ghpull:`18476`: CI: skip qt, cairo, pygobject related installs on OSX on travis
* :ghpull:`18134`: Build on xcode9

Issues (14):

* :ghissue:`18885`: 3D Scatter Plot with Colorbar is not saved correctly with savefig
* :ghissue:`18922`: pyplot.xticks(): Font property specification is not effective except 1st tick label.
* :ghissue:`18481`: "%matplotlib notebook" not working in firefox with matplotlib 3.3.1
* :ghissue:`18595`: Getting internal "MatplotlibDeprecationWarning: shading='flat' ..."
* :ghissue:`18743`:  from mpl 3.2.2 to 3.3.0 enormous increase in creation time
* :ghissue:`18317`: pcolormesh: shading='nearest' and non-monotonic coordinates
* :ghissue:`18758`: Using Axis.grid(visible=True) results in TypeError for multiple values for keyword argument
* :ghissue:`18638`: ``matplotlib>=3.3.2`` breaks ``ipywidgets.interact``
* :ghissue:`18337`: Error installing matplotlib-3.3.1 using pip due to old version of certifi on conda environment
* :ghissue:`18620`: wx backend assertion error with fig.canvas.timer.start()
* :ghissue:`18551`: test_transparent_markers[pdf] is broken on v3.3.x Travis macOS
* :ghissue:`18580`: Animation freezes in Jupyter notebook
* :ghissue:`18547`: pcolormesh x-axis with datetime broken for nearest shading
* :ghissue:`18539`: Error in Axes.redraw_in_frame in use of ExitStack: push() takes 2 positional arguments but 3 were given
