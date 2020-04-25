## PR Summary

<!--
Thank you so much for your PR! Please summarize the purpose of the PR here
using at least 1-2 sentences.

To help us review your pull request, make sure:

- The PR title descriptively summarizes the changes.
    - e.g. "Raise `ValueError` on non-numeric input to `set_xlim`"
    - Please don't use non-descriptive titles such as "Addresses issue #8576".
- The PR summary includes:
    - How the PR implements any changes.
    - Why these changes are necessary.
- The PR is not out of master, but out of a separate branch.
    - e.g. `your-user-name:non-numeric-xlim -> matplotlib:master`
- Optional: PR cross-links related issues.
-->

## PR Checklists

<!-- Feel free to delete any checkboxes that do not apply to this PR. -->

New code:

- [ ] has pytest style unit tests (and `pytest` passes)
- [ ] is [Flake 8](https://flake8.pycqa.org/en/latest/) compliant (run `flake8` on changed files to check)
- [ ] is documented, with examples if plot related

New documentation:

- [ ] is Sphinx and numpydoc compliant (the docs should [build](https://matplotlib.org/devel/documenting_mpl.html#building-the-docs) without error)
- [ ] conforms to Matplotlib style conventions (if you have `flake8-docstrings` and `pydocstyle<4` installed, run `flake8 --docstring-convention=all` on changed files to check).
<!--
- If you are contributing fixes to docstrings, please pay attention to
  https://matplotlib.org/devel/documenting_mpl.html#formatting. In particular,
  note the difference between using single backquotes, double backquotes, and
  asterisks in the markup.
-->

New features:

- [ ] have an entry in `doc/users/next_whats_new/` (follow instructions in `doc/users/next_whats_new/README.rst`)

API changes:

- [ ] are documented in `doc/api/api_changes_[NEXT_VERSION]` if API changed in a backward-incompatible way (follow instructions in `doc/api/api_changes_[NEXT_VERSION]/README.rst`)

<!--
If you have further questions:

- A more complete development guide is available at
  https://matplotlib.org/devdocs/devel/index.html.

- Help with Git and GitHub is available at
  https://matplotlib.org/devel/gitwash/development_workflow.html.

We understand that PRs can sometimes be overwhelming, especially as the
reviews start coming in. Please let us know if the reviews are unclear,
the recommended next step seems overly demanding, you would like help in
addressing a reviewer's comments, or you have been waiting too long to hear
back on your PR.
-->
