"""
Redirecting old docs to new location
====================================

If an rst file is moved or its content subsumed in a different file, it
is desirable to redirect the old file to the new or existing file. This
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
from sphinx.domains import Domain
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
    app.add_domain(RedirectFromDomain)
    app.connect("build-finished", _generate_redirects)

    metadata = {'parallel_read_safe': True}
    return metadata


class RedirectFromDomain(Domain):
    """
    The sole purpose of this domain is a parallel_read_safe data store for the
    redirects mapping.
    """
    name = 'redirect_from'
    label = 'redirect_from'

    @property
    def redirects(self):
        """The mapping of the redirectes."""
        return self.data.setdefault('redirects', {})

    def clear_doc(self, docnames):
        self.redirects.clear()

    def merge_domaindata(self, docnames, otherdata):
        for src, dst in otherdata['redirects'].items():
            if src not in self.redirects:
                self.redirects[src] = dst
            elif self.redirects[src] != dst:
                raise ValueError(
                    f"Inconsistent redirections from {src} to "
                    F"{self.redirects[src]} and {otherdata.redirects[src]}")


class RedirectFrom(Directive):
    required_arguments = 1

    def run(self):
        redirected_doc, = self.arguments
        env = self.app.env
        builder = self.app.builder
        domain = env.get_domain('redirect_from')
        current_doc = env.path2doc(self.state.document.current_source)
        redirected_reldoc, _ = env.relfn2path(redirected_doc, current_doc)
        if redirected_reldoc in domain.redirects:
            raise ValueError(
                f"{redirected_reldoc} is already noted as redirecting to "
                f"{domain.redirects[redirected_reldoc]}")
        domain.redirects[redirected_reldoc] = current_doc
        return []


def _generate_redirects(app, exception):
    builder = app.builder
    if builder.name != "html" or exception:
        return
    for k, v in app.env.get_domain('redirect_from').redirects.items():
        p = Path(app.outdir, k + builder.out_suffix)
        html = HTML_TEMPLATE.format(v=builder.get_relative_uri(k, v))
        if p.is_file():
            if p.read_text() != html:
                logger.warning(f'A redirect-from directive is trying to '
                               f'create {p}, but that file already exists '
                               f'(perhaps you need to run "make clean")')
        else:
            logger.info(f'making refresh html file: {k} redirect to {v}')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html)
