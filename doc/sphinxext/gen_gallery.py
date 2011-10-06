# generate a thumbnail gallery of examples
template = """\
{%% extends "layout.html" %%}
{%% set title = "Thumbnail gallery" %%}


{%% block body %%}

<h3>Click on any image to see full size image and source code</h3>
<br/>

%s
{%% endblock %%}
"""

import os, glob, re, sys, warnings
import matplotlib.image as image

multiimage = re.compile('(.*)_\d\d')

def make_thumbnail(args):
    image.thumbnail(args[0], args[1], 0.3)

def out_of_date(original, derived):
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)

def gen_gallery(app, doctree):
    if app.builder.name != 'html':
        return

    outdir = app.builder.outdir
    rootdir = 'plot_directive/mpl_examples'

    # images we want to skip for the gallery because they are an unusual
    # size that doesn't layout well in a table, or because they may be
    # redundant with other images or uninteresting
    skips = set([
        'mathtext_examples',
        'matshow_02',
        'matshow_03',
        'matplotlib_icon',
        ])

    data = []
    thumbnails = {}

    for subdir in ('api', 'pylab_examples', 'mplot3d', 'widgets', 'axes_grid' ):
        origdir = os.path.join('build', rootdir, subdir)
        thumbdir = os.path.join(outdir, rootdir, subdir, 'thumbnails')
        if not os.path.exists(thumbdir):
            os.makedirs(thumbdir)

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
            if m is None:
                pyfile = '%s.py'%basename
            else:
                basename = m.group(1)
                pyfile = '%s.py'%basename

            data.append((subdir, basename,
                         os.path.join(rootdir, subdir, 'thumbnails', filename)))

    link_template = """\
    <a href="%s"><img src="%s" border="0" alt="%s"/></a>
    """

    if len(data) == 0:
        warnings.warn("No thumbnails were found")

    rows = []
    for (subdir, basename, thumbfile) in data:
        if thumbfile is not None:
            link = 'examples/%s/%s.html'%(subdir, basename)
            rows.append(link_template%(link, thumbfile, basename))

    # Only write out the file if the contents have actually changed.
    # Otherwise, this triggers a full rebuild of the docs
    content = template%'\n'.join(rows)
    gallery_path = os.path.join(app.builder.srcdir, '_templates', 'gallery.html')
    if os.path.exists(gallery_path):
        fh = file(gallery_path, 'r')
        regenerate = fh.read() != content
        fh.close()
    else:
        regenerate = True
    if regenerate:
        fh = file(gallery_path, 'w')
        fh.write(content)
        fh.close()

    for key in app.builder.status_iterator(
        thumbnails.iterkeys(), "generating thumbnails... ",
        length=len(thumbnails)):
        image.thumbnail(key, thumbnails[key], 0.3)

def setup(app):
    app.connect('env-updated', gen_gallery)
