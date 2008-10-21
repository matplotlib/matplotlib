# generate a thumbnail gallery of examples
template = """\
{%% extends "layout.html" %%}
{%% set title = "Thumbnail gallery" %%}


{%% block body %%}

<h3>Click on any image to see full size image and source code</h3>
<br>

%s
{%% endblock %%}
"""

import os, glob, re

multiimage = re.compile('(.*)_\d\d')

pwd = os.getcwd()
os.chdir('..')

rootdir = '_static/plot_directive/mpl_examples'

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
for subdir in ('api', 'pylab_examples', 'widgets'):
    thisdir = os.path.join(rootdir,subdir)
    if not os.path.exists(thisdir):
        raise RuntimeError('Cannot find %s'%thisdir)
    thumbdir = os.path.join(thisdir, 'thumbnails')
    if not os.path.exists(thumbdir):
        raise RuntimeError('Cannot find thumbnail dir %s'%thumbdir)
    #print thumbdir

    # we search for pdfs here because there is one pdf for each
    # successful image build (2 pngs since one is high res) and the
    # mapping between py files and images is 1->many
    for pdffile in sorted(glob.glob(os.path.join(thisdir, '*.pdf'))):
        basepath, filename = os.path.split(pdffile)
        basename, ext = os.path.splitext(filename)
        print 'generating', subdir, basename

        if basename in skips:
            print '    skipping', basename
            continue
        pngfile = os.path.join(thisdir, '%s.png'%basename)
        thumbfile = os.path.join(thumbdir, '%s.png'%basename)
        if not os.path.exists(pngfile):
            pngfile = None
        if not os.path.exists(thumbfile):
            thumbfile = None

        m = multiimage.match(basename)
        if m is None:
            pyfile = '%s.py'%basename
        else:
            basename = m.group(1)
            pyfile = '%s.py'%basename

        print '    ', pyfile, filename, basename, ext

        print '    ', pyfile, pngfile, thumbfile
        data.append((subdir, thisdir, pyfile, basename, pngfile, thumbfile))

link_template = """\
<a href="%s"><img src="%s" border="0" alt="%s"/></a>
"""


rows = []
for (subdir, thisdir, pyfile, basename, pngfile, thumbfile) in data:
    if thumbfile is not None:
        link = 'examples/%s/%s.html'%(subdir, basename)
        rows.append(link_template%(link, thumbfile, basename))



os.chdir(pwd)
fh = file('gallery.html', 'w')
fh.write(template%'\n'.join(rows))
fh.close()

