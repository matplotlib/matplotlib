## PR Summary

<!--
Please provide at least a 1-2 sentence summary of the purpose of the PR.
-->


## PR Checklists

<!-- Feel free to delete any checkboxes that do not apply to this PR. -->

New code:

- [ ] Has Pytest style unit tests (and `pytest lib/matplotlib/tests` passes)
- [ ] Code is [Flake 8](http://flake8.pycqa.org/en/latest/) compliant (run `flake8` on changed files to check)
- [ ] Code is documented, with examples if plot related

New documentation:

- [ ] Documentation is sphinx and numpydoc compliant (the docs should [build](https://matplotlib.org/devel/documenting_mpl.html#building-the-docs) without error)
- [ ] Documentation conforms to matplotlib style conventions. (If you have `flake8-docstrings` and `pydocstyle<4` installed, run `flake8 --docstring-convention=all` on changed files to check).
<!--
- If you are contributing fixes to docstrings, please pay attention to
  http://matplotlib.org/devel/documenting_mpl.html#formatting. In particular,
  note the difference between using single backquotes, double backquotes, and
  asterisks in the markup.
-->

New features:

- [ ] Have an entry in `doc/users/next_whats_new/` (follow instructions in `doc/users/next_whats_new/README.rst`)

API changes:

- [ ] Documented in `doc/api/api_changes_[NEXT_VERSION]` if API changed in a backward-incompatible way (follow instructions in `doc/api/api_changes_[NEXT_VERSION]/README.rst`)

<!--
Meta:

- PR title descriptively summarizes the changes. (For example, prefer "Raise `ValueError` on non-numeric input to `set_xlim`" to "Addresses issue #8576").
- PR has at least a 1-2 sentence summary.
- PR is not out of master, but out of a separate branch (e.g. `your-user-name:non-numeric-xlim -> matplotlib:master`)
- Optional: PR cross-links related issues.
-->

<!--
If you have further questions:

- A more complete development guide is available at
  https://matplotlib.org/devdocs/devel/index.html.

- Help with git and github is available at
  https://matplotlib.org/devel/gitwash/development_workflow.html.

We understand that PRs can sometimes be overwhelming, especially as the
reviews start coming in. Please let us know if the reviews are unclear,  
the recommended next step seems overly demanding, you would like help in
addressing a reviewer's comments, or you have been waiting too long to hear
back on your PR.
-->
