# -*- coding: utf-8 -*-
import codecs
import os
import re
import glob
import warnings

import sphinx.errors

import matplotlib.image as image


exclude_example_sections = ['units']
multiimage = re.compile('(.*?)(_\d\d){1,2}')

# generate a thumbnail gallery of examples
gallery_template = u"""\
{{% extends "layout.html" %}}
{{% set title = "Thumbnail gallery" %}}


{{% block body %}}

<h3>Click on any image to see full size image and source code</h3>
<br/>

<li><a class="reference internal" href="#">Gallery</a>
    <ul>
    {toc}
    </ul>
</li>

{gallery}

{{% endblock %}}
"""

header_template = u"""\
<div class="section" id="{section}">
<h4>
    {title}<a class="headerlink" href="#{section}" title="Permalink to this headline">¶</a>
</h4>"""

link_template = """\
<figure>
  <a href="{link}"><img src="{thumb}" border="0" alt="{basename}"/></a><br>
  <figcaption><a href="{link}">{title}</a></figcaption>
</figure>
"""

toc_template = u"""\
<li><a class="reference internal" href="#{section}">{title}</a></li>"""


def make_thumbnail(args):
    image.thumbnail(args[0], args[1], 0.3)


def out_of_date(original, derived):
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)


def gen_gallery(app, doctree):
    if app.builder.name not in ('html', 'htmlhelp'):
        return

    outdir = app.builder.outdir
    rootdir = 'plot_directive/mpl_examples'

    example_sections = list(app.builder.config.mpl_example_sections)
    for i, (subdir, title) in enumerate(example_sections):
        if subdir in exclude_example_sections:
            example_sections.pop(i)

    # images we want to skip for the gallery because they are an unusual
    # size that doesn't layout well in a table, or because they may be
    # redundant with other images or uninteresting
    skips = {'mathtext_examples',
             'matshow_02',
             'matshow_03',
             'matplotlib_icon'}

    thumbnails = {}
    rows = []
    toc_rows = []

    for subdir, title in example_sections:
        rows.append(header_template.format(title=title, section=subdir))
        toc_rows.append(toc_template.format(title=title, section=subdir))

        origdir = os.path.join('build', rootdir, subdir)
        thumbdir = os.path.join(outdir, rootdir, subdir, 'thumbnails')
        if not os.path.exists(thumbdir):
            os.makedirs(thumbdir)

        data = []

        for filename in sorted(glob.glob(os.path.join(origdir, '*.png'))):
            if filename.endswith("hires.png"):
                continue

            path, filename = os.path.split(filename)
            basename, ext = os.path.splitext(filename)
            if basename in skips:
                continue

            # Create thumbnails based on images in tmpdir, and place
            # them within the build tree
            orig_path = str(os.path.join(origdir, filename))
            thumb_path = str(os.path.join(thumbdir, filename))
            if out_of_date(orig_path, thumb_path) or True:
                thumbnails[orig_path] = thumb_path

            m = multiimage.match(basename)
            if m is not None:
                basename = m.group(1)

            data.append((subdir, basename,
                         os.path.join(rootdir, subdir, 'thumbnails', filename)))

        for (subdir, basename, thumbfile) in data:
            if thumbfile is not None:
                link = 'examples/%s/%s.html'%(subdir, basename)
                rows.append(link_template.format(link=link,
                                                 thumb=thumbfile,
                                                 basename=basename,
                                                 title=basename))

        if len(data) == 0:
            warnings.warn("No thumbnails were found in %s" % subdir)

        # Close out the <div> opened up at the top of this loop
        rows.append(u"</div>")

    content = gallery_template.format(toc=u'\n'.join(toc_rows),
                                      gallery=u'\n'.join(rows))

    # Only write out the file if the contents have actually changed.
    # Otherwise, this triggers a full rebuild of the docs

    gallery_path = os.path.join(app.builder.srcdir,
                                '_templates', 'gallery.html')
    if os.path.exists(gallery_path):
        with codecs.open(gallery_path, 'r', encoding='utf-8') as fh:
            regenerate = fh.read() != content
    else:
        regenerate = True

    if regenerate:
        with codecs.open(gallery_path, 'w', encoding='utf-8') as fh:
            fh.write(content)

    for key in app.builder.status_iterator(
            iter(thumbnails), "generating thumbnails... ",
            length=len(thumbnails)):
        if out_of_date(key, thumbnails[key]):
            image.thumbnail(key, thumbnails[key], 0.3)


def setup(app):
    app.connect('env-updated', gen_gallery)

    try: # multiple plugins may use mpl_example_sections
        app.add_config_value('mpl_example_sections', [], True)
    except sphinx.errors.ExtensionError:
        pass # mpl_example_sections already defined

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
