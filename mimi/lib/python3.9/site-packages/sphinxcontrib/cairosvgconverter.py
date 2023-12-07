# -*- coding: utf-8 -*-
"""
    sphinxcontrib.cairosvgconverter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Converts SVG images to PDF using CairoSVG in case the builder does not
    support SVG images natively (e.g. LaTeX).

    See <https://cairosvg.org/>.

    :copyright: Copyright 2018-2023 by Stefan Wiehler
                <sphinx_contribute@missinglinkelectronics.com> and
                Copyright 2020 by Marko Kohtala
                <marko.kohtala@gmail.com>.
    :license: BSD, see LICENSE.txt for details.
"""

from sphinx.errors import ExtensionError
from sphinx.locale import __
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging
from urllib.error import URLError

if False:
    # For type annotation
    from typing import Any, Dict  # NOQA
    from sphinx.application import Sphinx  # NOQA


logger = logging.getLogger(__name__)


class CairoSVGConverter(ImageConverter):
    conversion_rules = [
        ('image/svg+xml', 'application/pdf'),
    ]

    def is_available(self):
        # type: () -> bool
        """Confirms if CairoSVG package is available or not."""
        try:
            import cairosvg  # noqa: F401
            return True
        except ImportError:
            logger.warning(__('CairoSVG package cannot be imported. '
                              'Check if CairoSVG has been installed properly'))
            return False

    def convert(self, _from, _to):
        # type: (unicode, unicode) -> bool
        """Converts the image from SVG to PDF via CairoSVG."""
        import cairosvg
        try:
            cairosvg.svg2pdf(file_obj=open(_from, 'rb'), write_to=_to)
        except (OSError, URLError) as err:
            raise ExtensionError(__('CairoSVG converter failed with reason: '
                                    '%s') % err.reason)

        return True


def setup(app):
    # type: (Sphinx) -> Dict[unicode, Any]
    app.add_post_transform(CairoSVGConverter)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
