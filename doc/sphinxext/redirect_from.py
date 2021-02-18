"""
Redirecting old docs to new location
====================================

If an rst file is moved or its content subsumed in a different file, it
is desireable to redirect the old file to the new or existing file. This
extension enables this with a simple html refresh.

For example suppose ``doc/topic/old-page.rst`` is removed and its content
included in ``doc/topic/new-page.rst``.  We use the ``redirect-from``
directive in ``doc/topic/new-page.rst``::

    .. redirect-from:: /topic/old-page

This creates in the build directory a file ``build/html/topic/old-page.html``
that contains a relative refresh::

    <html>
      <head>
        <meta http-equiv="refresh" content="0; url=new-page.html">
      </head>
    </html>

If you need to redirect across subdirectory trees, that works as well.  For
instance if ``doc/topic/subdir1/old-page.rst`` is now found at
``doc/topic/subdir2/new-page.rst`` then ``new-page.rst`` just lists the
full path::

    .. redirect-from:: /topic/subdir1/old-page.rst

"""

from pathlib import Path
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """<html>
  <head>
    <meta http-equiv="refresh" content="0; url={v}">
  </head>
</html>
"""


def setup(app):
    RedirectFrom.app = app
    app.add_directive("redirect-from", RedirectFrom)
    app.connect("build-finished", _generate_redirects)


class RedirectFrom(Directive):
    required_arguments = 1
    redirects = {}

    def run(self):
        redirected_doc, = self.arguments
        env = self.app.env
        builder = self.app.builder
        current_doc = env.path2doc(self.state.document.current_source)
        redirected_reldoc, _ = env.relfn2path(redirected_doc, current_doc)
        if redirected_reldoc in self.redirects:
            raise ValueError(
                f"{redirected_reldoc} is already noted as redirecting to "
                f"{self.redirects[redirected_reldoc]}")
        self.redirects[redirected_reldoc] = builder.get_relative_uri(
            redirected_reldoc, current_doc)
        return []


def _generate_redirects(app, exception):
    builder = app.builder
    if builder.name != "html" or exception:
        return
    for k, v in RedirectFrom.redirects.items():
        p = Path(app.outdir, k + builder.out_suffix)
        if p.is_file():
            logger.warning(f'A redirect-from directive is trying to create '
                           f'{p}, but that file already exists (perhaps '
                           f'you need to run "make clean")')
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("x") as file:
                logger.info(f'making refresh html file: {k} redirect to {v}')
                file.write(HTML_TEMPLATE.format(v=v))
