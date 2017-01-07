from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from pprint import pprint
from nose.plugins import Plugin
import os

from ... import conversion_cache as ccache


class ConversionCache(Plugin):
    enabled = True
    name = 'conversion-cache'

    def options(self, parser, env=os.environ):
        super(ConversionCache, self).configure(parser, env)
        parser.add_option("--conversion-cache-max-size", action="store",
                          dest="conversion_cache_max_size",
                          help="conversion cache maximum size in bytes")
        parser.add_option("--conversion-cache-report-misses",
                          action="store_true",
                          dest="conversion_cache_report_misses",
                          help="report conversion cache misses")

    def configure(self, options, conf):
        super(ConversionCache, self).configure(options, conf)
        if self.enabled:
            max_size = options.conversion_cache_max_size
            self.report_misses = options.conversion_cache_report_misses
            if max_size is not None:
                ccache.conversion_cache = ccache.ConversionCache(
                    max_size=int(max_size))
            else:
                ccache.conversion_cache = ccache.ConversionCache()

    def report(self, stream):
        ccache.conversion_cache.expire()
        data = ccache.conversion_cache.report()
        print("Image conversion cache hit rate: %d/%d" %
              (len(data['hits']), len(data['gets'])), file=stream)
        if self.report_misses:
            print("Missed files:", file=stream)
            for filename in sorted(data['gets'].difference(data['hits'])):
                print("  %s" % filename, file=stream)
