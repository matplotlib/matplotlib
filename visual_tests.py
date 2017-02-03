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


html_template = """<html><head><style media="screen" type="text/css">
img{{
  width:100%;
  max-width:800px;
}}
</style>
</head><body>
{failed}
{body}
</body></html>
"""

subdir_template = """<h2>{subdir}</h2><table>
<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>
{rows}
</table>
"""

failed_template = """<h2>Only Failed</h2><table>
<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>
{rows}
</table>
"""

row_template = ('<tr>'
                '<td>{0}{1}</td>'
                '<td>{2}</td>'
                '<td><a href="{3}"><img src="{3}"></a></td>'
                '<td>{4}</td>'
                '</tr>')

linked_image_template = '<a href="{0}"><img src="{0}"></a>'


def run():
    # Build a website for visual comparison
    image_dir = "result_images"
    # build the website
    _subdirs = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
    # loop over all pictures
    _has_failure = False
    failed_rows = []
    body_sections = []
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

        subdir_rows = []
        for name, test in six.iteritems(pictures):
            expected_image = test.get('e', '')
            actual_image = test.get('c', '')

            if 'f' in test:
                # a real failure in the image generation, resulting in different images
                _has_failure = True
                status = " (failed)"
                failed = '<a href="{0}">diff</a>'.format(test['f'])
                current = linked_image_template.format(actual_image)
                failed_rows.append(row_template.format(name, "", current,
                                                       expected_image, failed))
            elif 'c' not in test:
                # A failure in the test, resulting in no current image
                _has_failure = True
                status = " (failed)"
                failed = '--'
                current = '(Failure in test, no image produced)'
                failed_rows.append(row_template.format(name, "", current,
                                                       expected_image, failed))
            else:
                status = " (passed)"
                failed = '--'
                current = linked_image_template.format(actual_image)
            subdir_rows.append(row_template.format(name, status, current,
                                                   expected_image, failed))

        body_sections.append(
            subdir_template.format(subdir=subdir, rows='\n'.join(subdir_rows)))

    if _has_failure:
        failed = failed_template.format(rows='\n'.join(failed_rows))
    else:
        failed = ''
    body = ''.join(body_sections)
    html = html_template.format(failed=failed, body=body)
    index = os.path.join(image_dir, "index.html")
    with open(index, "w") as f:
        f.write(html)

    try:
        import webbrowser
        webbrowser.open(index)
    except:
        print("Open {} in a browser for a visual comparison.".format(index))

if __name__ == '__main__':
    run()
