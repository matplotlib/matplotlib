All ``Arist`` now carry wai-aria data
-------------------------------------

It is now possible to attach `wai-aria
<https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles>`__ role
information to any `~matplotlib.artist.Aritist`.  These roles are industry the
industry standard for providing accessibility mark up on the web.  This
information can be used by downstream applications for providing accessible
descriptions of visualizations.  Best practices in the space are still
developing, but by providing a mechanism to store and access this information
we will enable this development.

There are three methods provided:

- `~matplotlib.artist.Aritist.set_aria` which will completely replace any existing roles.
- `~matplotlib.artist.Aritist.update_aria` which will update the current roles in-place.
- `~matplotlib.artist.Aritist.get_aria` which will return a copy of the current roles.

We currently do no validation on either the keys or the values.


Matplotlib will use the `'aria-label'` role when saving svg output if it is
provided.
