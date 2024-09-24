# -*- coding: utf-8 -*-
"""
    sphinxcontrib.rsvgconverter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Converts SVG images to PDF using libRSVG in case the builder does not
    support SVG images natively (e.g. LaTeX).

    :copyright: Copyright 2018-2023 by Stefan Wiehler
                <sphinx_contribute@missinglinkelectronics.com>.
    :license: BSD, see LICENSE.txt for details.
"""
import subprocess

from sphinx.errors import ExtensionError
from sphinx.locale import __
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging
from errno import ENOENT, EPIPE, EINVAL

if False:
    # For type annotation
    from typing import Any, Dict  # NOQA
    from sphinx.application import Sphinx  # NOQA


logger = logging.getLogger(__name__)


class RSVGConverter(ImageConverter):
    conversion_rules = [
        ('image/svg+xml', 'application/pdf'),
    ]

    def is_available(self):
        # type: () -> bool
        """Confirms if RSVG converter is available or not."""
        try:
            args = [self.config.rsvg_converter_bin, '--version']
            logger.debug('Invoking %r ...', args)
            ret = subprocess.call(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            if ret == 0:
                return True
            else:
                return False
        except (OSError, IOError):
            logger.warning(__('RSVG converter command %r cannot be run. '
                              'Check the rsvg_converter_bin setting'),
                           self.config.rsvg_converter_bin)
            return False

    def convert(self, _from, _to):
        # type: (unicode, unicode) -> bool
        """Converts the image from SVG to PDF via libRSVG."""
        try:
            args = ([self.config.rsvg_converter_bin] +
                    self.config.rsvg_converter_args +
                    ['--format=' + self.config.rsvg_converter_format,
                     '--output=' + _to, _from])
            logger.debug('Invoking %r ...', args)
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except OSError as err:
            if err.errno != ENOENT:  # No such file or directory
                raise
            logger.warning(__('RSVG converter command %r cannot be run. '
                              'Check the rsvg_converter_bin setting'),
                           self.config.rsvg_converter_bin)
            return False

        try:
            stdout, stderr = p.communicate()
        except (OSError, IOError) as err:
            if err.errno not in (EPIPE, EINVAL):
                raise
            stdout, stderr = p.stdout.read(), p.stderr.read()
            p.wait()
        if p.returncode != 0:
            raise ExtensionError(__('RSVG converter exited with error:\n'
                                    '[stderr]\n%s\n[stdout]\n%s') %
                                 (stderr, stdout))

        return True


def setup(app):
    # type: (Sphinx) -> Dict[unicode, Any]
    app.add_post_transform(RSVGConverter)
    app.add_config_value('rsvg_converter_bin', 'rsvg-convert', 'env')
    app.add_config_value('rsvg_converter_args', [], 'env')
    app.add_config_value('rsvg_converter_format', 'pdf', 'env')

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
