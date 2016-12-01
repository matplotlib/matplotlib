#!/usr/bin/env python
#
# This builds a html page of all images from the image comparison tests
# and opens that page in the browser.
#
#   $ python visual_tests.py
#

import os
import time
import six

from collections import defaultdict

def run():
    # Build a website for visual comparison
    image_dir = "result_images"
    # build the website
    _html = ""
    _html += """<html><head><style media="screen" type="text/css">
    img{
        width:100%;
        max-width:800px;
    }
    </style>
    </head><body>\n"""
    _subdirs = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
    # loop over all pictures
    _row = '<tr><td>{0} {1}</td><td>{2}</td><td><a href="{3}"><img src="{3}"></a></td><td>{4}</td>\n'
    _failed = ""
    _failed += "<h2>Only Failed</h2>"
    _failed += "<table>\n<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>\n"
    _has_failure = False
    _body = ""
    for subdir in _subdirs:
        if subdir == "test_compare_images":
            # these are the image which test the image comparison functions...
            continue
        pictures = defaultdict(dict)
        for file in os.listdir(os.path.join(image_dir, subdir)):
            if os.path.isdir(os.path.join(image_dir, subdir, file)):
                continue
            fn, fext = os.path.splitext(file)
            if fext != ".png":
                continue
            # Always use / for URLs.
            if "-failed-diff" in fn:
                pictures[fn[:-12]]["f"] = "/".join((subdir, file))
            elif "-expected" in fn:
                pictures[fn[:-9]]["e"] = "/".join((subdir, file))
            else:
                pictures[fn]["c"] = "/".join((subdir, file))

        _body += "<h2>{0}</h2>".format(subdir)
        _body += "<table>\n<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>\n"
        for name, test in six.iteritems(pictures):
            if test.get("f", None):
                # a real failure in the image generation, resulting in different images
                _has_failure = True
                s = "(failed)"
                failed = '<a href="{0}">diff</a>'.format(test.get("f", ""))
                current = '<a href="{0}"><img src="{0}"></a>'.format(test.get("c", ""))
                _failed += _row.format(name, "", current, test.get("e", ""), failed)
            elif test.get("c", None) is None:
                # A failure in the test, resulting in no current image
                _has_failure = True
                s = "(failed)"
                failed = '--'
                current = '(Failure in test, no image produced)'
                _failed += _row.format(name, "", current, test.get("e", ""), failed)
            else:
                s = "(passed)"
                failed = '--'
                current = '<a href="{0}"><img src="{0}"></a>'.format(test.get("c", ""))
            _body += _row.format(name, "", current, test.get("e", ""), failed)
        _body += "</table>\n"
    _failed += "</table>\n"
    if _has_failure:
        _html += _failed
    _html += _body
    _html += "\n</body></html>"
    index = os.path.join(image_dir, "index.html")
    with open(index, "w") as f:
        f.write(_html)
    try:
        import webbrowser
        webbrowser.open(index)
    except:
        print("Open {} in a browser for a visual comparison.".format(index))

if __name__ == '__main__':
    run()
