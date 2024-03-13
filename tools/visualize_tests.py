#!/usr/bin/env python
#
# This builds a html page of all images from the image comparison tests
# and opens that page in the browser.
#
#   $ python tools/visualize_tests.py
#

import argparse
import os
from collections import defaultdict

# Non-png image extensions
NON_PNG_EXTENSIONS = ['pdf', 'svg', 'eps']

html_template = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Matplotlib test result visualization</title>
<style media="screen">
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
<thead><tr><th>name</th><th>actual</th><th>expected</th><th>diff</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

failed_template = """<h2>Only Failed</h2><table>
<thead><tr><th>name</th><th>actual</th><th>expected</th><th>diff</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

row_template = ('<tr>'
                '<td>{0}{1}</td>'
                '<td>{2}</td>'
                '<td><a href="{3}"><img src="{3}"></a></td>'
                '<td>{4}</td>'
                '</tr>')

linked_image_template = '<a href="{0}"><img src="{0}"></a>'


def run(show_browser=True):
    """
    Build a website for visual comparison
    """
    image_dir = "result_images"
    _subdirs = (name
                for name in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, name)))

    failed_rows = []
    body_sections = []
    for subdir in sorted(_subdirs):
        if subdir == "test_compare_images":
            # These are the images which test the image comparison functions.
            continue

        pictures = defaultdict(dict)
        for file in os.listdir(os.path.join(image_dir, subdir)):
            if os.path.isdir(os.path.join(image_dir, subdir, file)):
                continue
            fn, fext = os.path.splitext(file)
            if fext != ".png":
                continue
            if "-failed-diff" in fn:
                file_type = 'diff'
                test_name = fn[:-len('-failed-diff')]
            elif "-expected" in fn:
                for ext in NON_PNG_EXTENSIONS:
                    if fn.endswith(f'_{ext}'):
                        display_extension = f'_{ext}'
                        extension = ext
                        fn = fn[:-len(display_extension)]
                        break
                else:
                    display_extension = ''
                    extension = 'png'
                file_type = 'expected'
                test_name = fn[:-len('-expected')] + display_extension
            else:
                file_type = 'actual'
                test_name = fn
            # Always use / for URLs.
            pictures[test_name][file_type] = '/'.join((subdir, file))

        subdir_rows = []
        for name, test in sorted(pictures.items()):
            expected_image = test.get('expected', '')
            actual_image = test.get('actual', '')

            if 'diff' in test:
                # A real failure in the image generation, resulting in
                # different images.
                status = " (failed)"
                failed = f'<a href="{test["diff"]}">diff</a>'
                current = linked_image_template.format(actual_image)
                failed_rows.append(row_template.format(name, "", current,
                                                       expected_image, failed))
            elif 'actual' not in test:
                # A failure in the test, resulting in no current image
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

    if failed_rows:
        failed = failed_template.format(rows='\n'.join(failed_rows))
    else:
        failed = ''
    body = ''.join(body_sections)
    html = html_template.format(failed=failed, body=body)
    index = os.path.join(image_dir, "index.html")
    with open(index, "w") as f:
        f.write(html)

    show_message = not show_browser
    if show_browser:
        try:
            import webbrowser
            webbrowser.open(index)
        except Exception:
            show_message = True

    if show_message:
        print(f"Open {index} in a browser for a visual comparison.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true',
                        help="Don't show browser after creating index page.")
    args = parser.parse_args()
    run(show_browser=not args.no_browser)
