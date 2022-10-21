.. _github-stats-3-2-1:

GitHub statistics for 3.2.1 (Mar 18, 2020)
==========================================

GitHub statistics for 2020/03/03 (tag: v3.2.0) - 2020/03/18

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed 11 issues and merged 52 pull requests.
The full list can be seen `on GitHub <https://github.com/matplotlib/matplotlib/milestone/49?closed=1>`__
and `on GitHub <https://github.com/matplotlib/matplotlib/milestone/51?closed=1>`__

The following 12 authors contributed 154 commits.

* Amy Roberts
* Antony Lee
* Elliott Sales de Andrade
* hannah
* Hugo van Kemenade
* Jody Klymak
* Kyle Sunden
* MarcoGorelli
* Maximilian Nöthe
* Sandro Tosi
* Thomas A Caswell
* Tim Hoffmann

GitHub issues and pull requests:

Pull Requests (52):

* :ghpull:`15199`: MNT/TST: generalize check_figures_equal to work with pytest.marks
* :ghpull:`15685`: Avoid a RuntimeError at animation shutdown with PySide2.
* :ghpull:`15969`: Restart pgf's latex instance after bad latex inputs.
* :ghpull:`16640`: ci: Fix Azure on v3.2.x
* :ghpull:`16648`: Document filling of Poly3DCollection
* :ghpull:`16649`: Fix typo in docs
* :ghpull:`16650`: Backport PR #16649 on branch v3.2.x (Fix typo in docs)
* :ghpull:`16651`: Docs: Change Python 2 note to past tense
* :ghpull:`16654`: Backport PR #16651 on branch v3.2.0-doc (Docs: Change Python 2 note to past tense)
* :ghpull:`16656`: Make test_imagegrid_cbar_mode_edge less flaky.
* :ghpull:`16661`: added Framework :: Matplotlib  to setup
* :ghpull:`16665`: Backport PR #16661 on branch v3.2.x (added Framework :: Matplotlib  to setup)
* :ghpull:`16671`: Fix some readme bits
* :ghpull:`16672`: Update CircleCI and add direct artifact link
* :ghpull:`16682`: Avoid floating point rounding causing bezier.get_parallels to fail
* :ghpull:`16690`: Backport PR #16682 on branch v3.2.x (Avoid floating point rounding causing bezier.get_parallels to fail)
* :ghpull:`16693`: TST: use pytest name in naming files for check_figures_equal
* :ghpull:`16695`: Restart pgf's latex instance after bad latex inputs.
* :ghpull:`16705`: Backport PR #16656 on branch v3.2.x (Make test_imagegrid_cbar_mode_edge less flaky.)
* :ghpull:`16708`: Backport PR #16671: Fix some readme bits
* :ghpull:`16709`: Fix saving PNGs to file objects in some places
* :ghpull:`16722`: Deprecate rcParams["datapath"] in favor of mpl.get_data_path().
* :ghpull:`16725`: TST/CI: also try to run test_user_fonts_win32 on azure
* :ghpull:`16734`: Disable draw_foo methods on renderer used to estimate tight extents.
* :ghpull:`16735`: Make test_stem less flaky.
* :ghpull:`16736`: xpdf: Set AutoRotatePages to None, not false.
* :ghpull:`16742`: nbagg: Don't send events if manager is disconnected.
* :ghpull:`16745`: Allow numbers to set uvc for all arrows in quiver.set_UVC, fixes #16743
* :ghpull:`16751`: Backport PR #16742 on branch v3.2.x (nbagg: Don't send events if manager is disconnected.)
* :ghpull:`16752`: ci: Disallow pytest 5.4.0, which is crashing.
* :ghpull:`16753`: Backport #16752 to v3.2.x
* :ghpull:`16760`: Backport PR #16735 on branch v3.2.x (Make test_stem less flaky.)
* :ghpull:`16761`: Backport PR #16745 on branch v3.2.x (Allow numbers to set uvc for all arrows in quiver.set_UVC, fixes #16743)
* :ghpull:`16763`: Backport PR #16648 on branch v3.2.x (Document filling of Poly3DCollection)
* :ghpull:`16764`: Backport PR #16672 on branch v3.2.0-doc
* :ghpull:`16765`: Backport PR #16736 on branch v3.2.x (xpdf: Set AutoRotatePages to None, not false.)
* :ghpull:`16766`: Backport PR #16734 on branch v3.2.x (Disable draw_foo methods on renderer used to estimate tight extents.)
* :ghpull:`16767`: Backport PR #15685 on branch v3.2.x (Avoid a RuntimeError at animation shutdown with PySide2.)
* :ghpull:`16768`: Backport PR #16725 on branch v3.2.x (TST/CI: also try to run test_user_fonts_win32 on azure)
* :ghpull:`16770`: Fix tuple markers
* :ghpull:`16779`: Documentation: make instructions for documentation contributions easier to find, add to requirements for building docs
* :ghpull:`16784`: Update CircleCI URL for downloading humor-sans.ttf.
* :ghpull:`16790`: Backport PR #16784 on branch v3.2.x (Update CircleCI URL for downloading humor-sans.ttf.)
* :ghpull:`16791`: Backport PR #16770 on branch v3.2.x (Fix tuple markers)
* :ghpull:`16794`: DOC: Don't mention drawstyle in ``set_linestyle`` docs.
* :ghpull:`16795`: Backport PR #15199 on branch v3.2.x (MNT/TST: generalize check_figures_equal to work with pytest.marks)
* :ghpull:`16797`: Backport #15589 and #16693, fixes for check_figures_equal
* :ghpull:`16799`: Backport PR #16794 on branch v3.2.0-doc (DOC: Don't mention drawstyle in ``set_linestyle`` docs.)
* :ghpull:`16800`: Fix check_figures_equal for tests that use its fixtures.
* :ghpull:`16803`: Fix some doc issues
* :ghpull:`16806`: Backport PR #16803 on branch v3.2.0-doc (Fix some doc issues)
* :ghpull:`16809`: Backport PR #16779 on branch v3.2.0-doc (Documentation: make instructions for documentation contributions easier to find, add to requirements for building docs)

Issues (11):

* :ghissue:`12820`: [Annotations] ValueError: lines do not intersect when computing tight bounding box containing arrow with filled paths
* :ghissue:`16538`: xpdf distiller seems broken
* :ghissue:`16624`: Azure pipelines are broken on v3.2.x
* :ghissue:`16633`: Wrong drawing Poly3DCollection
* :ghissue:`16645`: Minor typo in API document of patches.ConnectionPatch
* :ghissue:`16670`: BLD: ascii codec decode on 3.2.0 in non-UTF8 locales
* :ghissue:`16704`: 3.2.0: ``setup.py clean`` fails with ``NameError: name 'long_description' is not defined``
* :ghissue:`16721`: nbAgg backend does not allow saving figures as png
* :ghissue:`16731`: PGF backend + savefig.bbox results in I/O error in 3.2
* :ghissue:`16743`: Breaking change in 3.2: quiver.set_UVC does not support single numbers any more
* :ghissue:`16801`: Doc: figure for colormaps off
