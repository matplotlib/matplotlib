[33mcommit 156001ba7cc589f2bdd1e6820973e8e1595fde53[m[33m ([m[1;36mHEAD -> [m[1;32mlog-python-warnings-plot-directive[m[33m, [m[1;31morigin/log-python-warnings-plot-directive[m[33m)[m
Author: Boyu Dai <u7241110@anu.edu.au>
Date:   Sun Jan 4 18:16:33 2026 +1100

    Lodge warning during plot to sphinx warning

[33mcommit d99834c15bf80d88eca14f98a03e4c25c4b6c678[m[33m ([m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m, [m[1;32mmain[m[33m)[m
Merge: 7863bdedf2 b9b5d627b1
Author: Elliott Sales de Andrade <quantum.analyst@gmail.com>
Date:   Sat Jan 3 18:38:28 2026 -0500

    Merge pull request #30921 from timhoffm/ci-stale-exclude

    Exclude confirmed bugs from stale bot

[33mcommit 7863bdedf22174c2727227faad0cbf1dfa098dbb[m
Merge: cddd97354b ab0247681c
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Sat Jan 3 19:23:19 2026 +0100

    Merge pull request #30892 from matplotlib/dependabot/github_actions/actions-1399cdcb1c

    Bump the actions group across 1 directory with 11 updates

[33mcommit cddd97354be753c85929a57c943dd012d8b62f21[m
Merge: 6688d5d1a5 e032e2524f
Author: Elliott Sales de Andrade <quantum.analyst@gmail.com>
Date:   Fri Jan 2 23:39:57 2026 -0500

    Merge pull request #30920 from codingabhiroop/fix_flaky_test_reruns

    FIX: Increase reruns for flaky test_invisible_Line_rendering (#30809)

[33mcommit b9b5d627b1d9a87cadafd842512bd3d730990df6[m
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Fri Jan 2 13:44:58 2026 +0100

    Exclude confirmed bugs from stale-tidy bot

    We should not close confirmed bugs through a timeout mechanism. The bug
    exists and should be tracked as open bug as long as we don't fix it or
    explicitly decide to not handle this, in which case, we'd manually close
     as "won't fix".

[33mcommit 6688d5d1a51ed8cd0c7d838681a39bf84a5f2fd0[m
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Fri Jan 2 21:57:18 2026 +0100

    DOC: Improve writer parameter docs of Animation.save() (#30910)

    Closes #24159.

[33mcommit a9ed0e1f98a279503bd08522a47afc741c70b190[m
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Fri Jan 2 21:55:48 2026 +0100

    MNT: Make transforms helper functions private (#30889)

    * MNT: Make transforms.nonsingular private

    * MNT: Make transforms.interval_contains_open private

    * MNT: Make transforms.interval_contains private

    * Update doc/api/next_api_changes/deprecations/30889-TH.rst

    Co-authored-by: Thomas A Caswell <tcaswell@gmail.com>

    * Copy docstring via decorator

    ... to play nicely with deprecation warnings in docstrings

    ---------

    Co-authored-by: Thomas A Caswell <tcaswell@gmail.com>

[33mcommit 7f8cde84c9159ba6a2d1f19b48e32e3071420306[m
Merge: b2dc2a6a56 789153aa20
Author: Jody Klymak <jklymak@gmail.com>
Date:   Fri Jan 2 09:47:47 2026 -0800

    Merge pull request #30922 from timhoffm/ci-stale-run-frequency

    Reduce stale bot to run once per week

[33mcommit 789153aa20ebcdcbfa4a5c376bd418c443dfccea[m
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Fri Jan 2 13:37:52 2026 +0100

    Reduce stale bot to run once per week

    I have the impression that we are not systematically reviewing the
    issues marked as inactive. We had multiple cases of re-opening issues,
    which means they haven't been identified as "keep" during the
    stale phase. IMHO this should not happen as it increases the danger of
    overlooking the
     closing of relevant issues.

[33mcommit b2dc2a6a56f7c089b34ece316109100a20bf2ed1[m
Author: Sanchit Rishi <sharmasanchitrishi@gmail.com>
Date:   Fri Jan 2 16:59:36 2026 +0530

    Pcolormesh Doc Fix (#30912)


    ---------

    Co-authored-by: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>

[33mcommit e032e2524f24e1dcb9315dcb31e72ee05bb0d99d[m
Author: ee25b003 <ee25b003@smail.iitm.ac.in>
Date:   Thu Jan 1 22:03:04 2026 +0530

    FIX: Increase reruns for flaky test_invisible_Line_rendering (#30809)

[33mcommit 0bf4d39580234710738dbef3f51b6770e1f61fac[m
Author: Abhiroop Batabyal <coding.abhiroop@gmail.com>
Date:   Thu Jan 1 15:52:11 2026 +0530

    Docs: Remove outdated annotate_transform example, link to annotation tutorial (#30916)

    * Docs: Remove outdated annotate_transform example, link to annotation tutorial instead

    * Docs: remove annotate_transform example and redirect to annotations tutorial

    * DOC: Add redirect for removed annotate_transform example

    ---------

    Co-authored-by: ee25b003 <ee25b003@smail.iitm.ac.in>

[33mcommit b9aaca35cb81b6a008683b1f9e9b3c160c52a68a[m
Merge: 776b24910f 75b27f8ce3
Author: Ruth Comer <10599679+rcomer@users.noreply.github.com>
Date:   Thu Jan 1 10:15:30 2026 +0000

    Merge pull request #30919 from star1327p/a-an-usage

    DOC: Correct typos on a/an usage including print messages

[33mcommit 75b27f8ce39c2b27f894b431b278b68756e6f1f0[m
Author: star1327p <star1327p@gmail.com>
Date:   Wed Dec 31 15:49:05 2025 -0800

    DOC: Correct typos on a/an usage including print messages

[33mcommit 776b24910f433d7dbd5bd9ee72d0c1f245eaa794[m
Merge: 0dc49d9794 babeefaccd
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Tue Dec 30 19:46:27 2025 +0100

    Merge pull request #30914 from codingabhiroop/doc-fix-violin-links

    Fix outdated documentation links for violin/boxplot example

[33mcommit babeefaccd62c6a07d277ed3c7aade2439c7dbb3[m
Author: ee25b003 <ee25b003@smail.iitm.ac.in>
Date:   Tue Dec 30 22:02:12 2025 +0530

    Fix outdated documentation links for violin/boxplot example

[33mcommit 0dc49d9794ab47776019af1595e7d58219b490e3[m
Merge: 7bbe3b7e19 786b06375d
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Tue Dec 30 02:52:15 2025 +0100

    Merge pull request #30907 from anntzer/aas

    Inline intermediate constructs in axisartist demos.

[33mcommit ab0247681c324f787a1d75c3a451c436b9f257d6[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Mon Dec 29 19:03:01 2025 +0000

    Bump the actions group across 1 directory with 11 updates

    Bumps the actions group with 11 updates in the / directory:

    | Package | From | To |
    | --- | --- | --- |
    | [actions/checkout](https://github.com/actions/checkout) | `5.0.0` | `6.0.1` |
    | [actions/setup-python](https://github.com/actions/setup-python) | `6.0.0` | `6.1.0` |
    | [actions/upload-artifact](https://github.com/actions/upload-artifact) | `5.0.0` | `6.0.0` |
    | [actions/download-artifact](https://github.com/actions/download-artifact) | `6.0.0` | `7.0.0` |
    | [pypa/cibuildwheel](https://github.com/pypa/cibuildwheel) | `3.2.1` | `3.3.0` |
    | [reviewdog/action-setup](https://github.com/reviewdog/action-setup) | `1.4.0` | `1.5.0` |
    | [github/codeql-action](https://github.com/github/codeql-action) | `4.31.0` | `4.31.9` |
    | [actions/cache](https://github.com/actions/cache) | `4.3.0` | `5.0.1` |
    | [scientific-python/upload-nightly-action](https://github.com/scientific-python/upload-nightly-action) | `0.6.2` | `0.6.3` |
    | [actions/stale](https://github.com/actions/stale) | `10.1.0` | `10.1.1` |
    | [codecov/codecov-action](https://github.com/codecov/codecov-action) | `5.5.1` | `5.5.2` |



    Updates `actions/checkout` from 5.0.0 to 6.0.1
    - [Release notes](https://github.com/actions/checkout/releases)
    - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
    - [Commits](https://github.com/actions/checkout/compare/08c6903cd8c0fde910a37f88322edcfb5dd907a8...8e8c483db84b4bee98b60c0593521ed34d9990e8)

    Updates `actions/setup-python` from 6.0.0 to 6.1.0
    - [Release notes](https://github.com/actions/setup-python/releases)
    - [Commits](https://github.com/actions/setup-python/compare/e797f83bcb11b83ae66e0230d6156d7c80228e7c...83679a892e2d95755f2dac6acb0bfd1e9ac5d548)

    Updates `actions/upload-artifact` from 5.0.0 to 6.0.0
    - [Release notes](https://github.com/actions/upload-artifact/releases)
    - [Commits](https://github.com/actions/upload-artifact/compare/330a01c490aca151604b8cf639adc76d48f6c5d4...b7c566a772e6b6bfb58ed0dc250532a479d7789f)

    Updates `actions/download-artifact` from 6.0.0 to 7.0.0
    - [Release notes](https://github.com/actions/download-artifact/releases)
    - [Commits](https://github.com/actions/download-artifact/compare/018cc2cf5baa6db3ef3c5f8a56943fffe632ef53...37930b1c2abaa49bbe596cd826c3c89aef350131)

    Updates `pypa/cibuildwheel` from 3.2.1 to 3.3.0
    - [Release notes](https://github.com/pypa/cibuildwheel/releases)
    - [Changelog](https://github.com/pypa/cibuildwheel/blob/main/docs/changelog.md)
    - [Commits](https://github.com/pypa/cibuildwheel/compare/9c00cb4f6b517705a3794b22395aedc36257242c...63fd63b352a9a8bdcc24791c9dbee952ee9a8abc)

    Updates `reviewdog/action-setup` from 1.4.0 to 1.5.0
    - [Release notes](https://github.com/reviewdog/action-setup/releases)
    - [Commits](https://github.com/reviewdog/action-setup/compare/d8edfce3dd5e1ec6978745e801f9c50b5ef80252...d8a7baabd7f3e8544ee4dbde3ee41d0011c3a93f)

    Updates `github/codeql-action` from 4.31.0 to 4.31.9
    - [Release notes](https://github.com/github/codeql-action/releases)
    - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
    - [Commits](https://github.com/github/codeql-action/compare/4e94bd11f71e507f7f87df81788dff88d1dacbfb...5d4e8d1aca955e8d8589aabd499c5cae939e33c7)

    Updates `actions/cache` from 4.3.0 to 5.0.1
    - [Release notes](https://github.com/actions/cache/releases)
    - [Changelog](https://github.com/actions/cache/blob/main/RELEASES.md)
    - [Commits](https://github.com/actions/cache/compare/0057852bfaa89a56745cba8c7296529d2fc39830...9255dc7a253b0ccc959486e2bca901246202afeb)

    Updates `scientific-python/upload-nightly-action` from 0.6.2 to 0.6.3
    - [Release notes](https://github.com/scientific-python/upload-nightly-action/releases)
    - [Commits](https://github.com/scientific-python/upload-nightly-action/compare/b36e8c0c10dbcfd2e05bf95f17ef8c14fd708dbf...5748273c71e2d8d3a61f3a11a16421c8954f9ecf)

    Updates `actions/stale` from 10.1.0 to 10.1.1
    - [Release notes](https://github.com/actions/stale/releases)
    - [Changelog](https://github.com/actions/stale/blob/main/CHANGELOG.md)
    - [Commits](https://github.com/actions/stale/compare/5f858e3efba33a5ca4407a664cc011ad407f2008...997185467fa4f803885201cee163a9f38240193d)

    Updates `codecov/codecov-action` from 5.5.1 to 5.5.2
    - [Release notes](https://github.com/codecov/codecov-action/releases)
    - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
    - [Commits](https://github.com/codecov/codecov-action/compare/5a1091511ad55cbe89839c7260b706298ca349f7...671740ac38dd9b0130fbe1cec585b89eea48d3de)

    ---
    updated-dependencies:
    - dependency-name: actions/checkout
      dependency-version: 6.0.1
      dependency-type: direct:production
      update-type: version-update:semver-major
      dependency-group: actions
    - dependency-name: actions/setup-python
      dependency-version: 6.1.0
      dependency-type: direct:production
      update-type: version-update:semver-minor
      dependency-group: actions
    - dependency-name: actions/upload-artifact
      dependency-version: 6.0.0
      dependency-type: direct:production
      update-type: version-update:semver-major
      dependency-group: actions
    - dependency-name: actions/download-artifact
      dependency-version: 7.0.0
      dependency-type: direct:production
      update-type: version-update:semver-major
      dependency-group: actions
    - dependency-name: pypa/cibuildwheel
      dependency-version: 3.3.0
      dependency-type: direct:production
      update-type: version-update:semver-minor
      dependency-group: actions
    - dependency-name: reviewdog/action-setup
      dependency-version: 1.5.0
      dependency-type: direct:production
      update-type: version-update:semver-minor
      dependency-group: actions
    - dependency-name: github/codeql-action
      dependency-version: 4.31.9
      dependency-type: direct:production
      update-type: version-update:semver-patch
      dependency-group: actions
    - dependency-name: actions/cache
      dependency-version: 5.0.1
      dependency-type: direct:production
      update-type: version-update:semver-major
      dependency-group: actions
    - dependency-name: scientific-python/upload-nightly-action
      dependency-version: 0.6.3
      dependency-type: direct:production
      update-type: version-update:semver-patch
      dependency-group: actions
    - dependency-name: actions/stale
      dependency-version: 10.1.1
      dependency-type: direct:production
      update-type: version-update:semver-patch
      dependency-group: actions
    - dependency-name: codecov/codecov-action
      dependency-version: 5.5.2
      dependency-type: direct:production
      update-type: version-update:semver-patch
      dependency-group: actions
    ...

    Signed-off-by: dependabot[bot] <support@github.com>

[33mcommit 786b06375ded4e3ca6e408fc1018e7d49993cdb9[m
Author: Antony Lee <anntzer.lee@gmail.com>
Date:   Mon Dec 29 14:23:31 2025 +0100

    Inline intermediate constructs in axisartist demos.

    A bunch of `grid_locator1=grid_locator1` kwargs passing doesn't really
    help legibility, in particular with respect to showing the object
    dependency hierarchy.

[33mcommit 7bbe3b7e19fc0de2c9461837cbd015a03f31b434[m
Author: Julian Chen <gapplef@gmail.com>
Date:   Sat Dec 27 19:16:11 2025 +0800

    Handle single color for multiple datasets in `hist` (#30867)

[33mcommit 15697eab459dce356f86e0507ddb44a8a24cfac6[m
Merge: bbf01d4216 bec8b2b077
Author: Thomas A Caswell <tcaswell@gmail.com>
Date:   Tue Dec 23 11:27:59 2025 -0500

    Merge pull request #30591 from timhoffm/safe-blit

    FIX: Make widget blitting compatible with swapped canvas

[33mcommit bbf01d4216560fd1a1c73422966e73cc38dfb1cf[m
Author: brk <187102275+brooks-code@users.noreply.github.com>
Date:   Fri Dec 19 20:34:09 2025 +0100

    Implements the Okabe-Ito accessible colormap. (#30821)

    Add Okabe_Ito color sequence (https://jfly.uni-koeln.de/color/)

[33mcommit 00881824d1f07d1c282d40346249a8ac41234d1a[m
Merge: 7b64c5584d d79825675c
Author: Eric Firing <efiring@hawaii.edu>
Date:   Thu Dec 18 10:54:13 2025 -1000

    Merge pull request #30737 from timhoffm/multicursor

    Deprecate unused canvas parameter to MultiCursor

[33mcommit d79825675cd492369ea33b645ab02afdf7fc9949[m
Author: Greg Lucas <greg.m.lucas@gmail.com>
Date:   Thu Dec 18 13:48:13 2025 -0700

    Update lib/matplotlib/widgets.py

    Co-authored-by: Elliott Sales de Andrade <quantum.analyst@gmail.com>

[33mcommit 7b64c5584d544f95c479f8d521b4af51646fed6f[m
Merge: 57ad96d45c 9bfaffef00
Author: Greg Lucas <greg.m.lucas@gmail.com>
Date:   Thu Dec 18 13:44:31 2025 -0700

    Merge pull request #29966 from anntzer/outset-widget

    Fix AxesWidgets on inset_axes that are outside their parent.

[33mcommit 57ad96d45c23e070331fa99547cf79321eb7c3c6[m
Author: Melwyn Francis Carlo <66683108+melwyncarlo@users.noreply.github.com>
Date:   Thu Dec 18 21:38:37 2025 +0530

    Implement warning for Text3D's rotation/rotation_mode parameters (#30600)


    Co-authored-by: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>

[33mcommit 3ba3d6f0ab77899ee9f8120eff395e81a0d3e973[m
Author: Elliott Sales de Andrade <quantum.analyst@gmail.com>
Date:   Wed Dec 17 18:14:07 2025 -0500

    Fix test_ensure_multivariate_data on 32-bit systems (#30847)

    In that case, the default int is also 32-bit, so the test will fail to
    be equal to `int64`.

    This is similar to #30629.

[33mcommit 90748a5669fc2d179bf39895b37e7572bbfb8e08[m
Author: Tim Hoffmann <2836374+timhoffm@users.noreply.github.com>
Date:   Wed Dec 17 00:47:12 2025 +0100

    DOC: Rectangle: Link to FancyBboxPatch for rounded corners (#30856)

    * DOC: Rectangle: Link to FancyBboxPatch for rounded corners

    Closes #27969.

    * Update lib/matplotlib/patches.py

    Co-authored-by: Ruth Comer <10599679+rcomer@users.noreply.github.com>

    ---------

    Co-authored-by: Ruth Comer <10599679+rcomer@users.noreply.github.com>
