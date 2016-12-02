"""
generate the rst files for the examples by iterating over the pylab examples
"""
from __future__ import print_function
import io
import os
import re
import sys

import sphinx.errors


exclude_example_sections = ['widgets']
noplot_regex = re.compile(r"#\s*-\*-\s*noplot\s*-\*-")


def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.

    TODO: this check isn't adequate in some cases.  e.g., if we discover
    a bug when building the examples, the original and derived will be
    unchanged but we still want to force a rebuild.
    """
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)

def generate_example_rst(app):
    rootdir = os.path.join(app.builder.srcdir, 'mpl_examples')
    exampledir = os.path.join(app.builder.srcdir, 'examples')
    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    example_sections = list(app.builder.config.mpl_example_sections)
    for i, (subdir, title) in enumerate(example_sections):
        if subdir in exclude_example_sections:
            example_sections.pop(i)
    example_subdirs, titles = zip(*example_sections)

    datad = {}
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if ( fname.startswith('.') or fname.startswith('#')
                 or fname.startswith('_') or not fname.endswith('.py') ):
                continue

            fullpath = os.path.join(root,fname)
            contents = io.open(fullpath, encoding='utf8').read()
            # indent
            relpath = os.path.split(root)[-1]
            datad.setdefault(relpath, []).append((fullpath, fname, contents))

    subdirs = sorted(datad)

    fhindex = open(os.path.join(exampledir, 'index.rst'), 'w')
    fhindex.write("""\
.. _examples-index:

####################
Matplotlib Examples
####################

.. htmlonly::

    :Release: |version|
    :Date: |today|

.. toctree::
    :maxdepth: 2

""")

    for subdir in subdirs:
        rstdir = os.path.join(exampledir, subdir)
        if not os.path.exists(rstdir):
            os.makedirs(rstdir)

        outputdir = os.path.join(app.builder.outdir, 'examples')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        outputdir = os.path.join(outputdir, subdir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        subdirIndexFile = os.path.join(rstdir, 'index.rst')
        fhsubdirIndex = open(subdirIndexFile, 'w')
        fhindex.write('    %s/index.rst\n\n'%subdir)

        fhsubdirIndex.write("""\
.. _%s-examples-index:

##############################################
%s Examples
##############################################

.. htmlonly::

    :Release: |version|
    :Date: |today|

.. toctree::
    :maxdepth: 1

"""%(subdir, subdir))

        sys.stdout.write(subdir + ", ")
        sys.stdout.flush()

        data = datad[subdir]
        data.sort()

        for fullpath, fname, contents in data:
            basename, ext = os.path.splitext(fname)
            outputfile = os.path.join(outputdir, fname)
            #thumbfile = os.path.join(thumb_dir, '%s.png'%basename)
            #print '    static_dir=%s, basename=%s, fullpath=%s, fname=%s, thumb_dir=%s, thumbfile=%s'%(static_dir, basename, fullpath, fname, thumb_dir, thumbfile)

            rstfile = '%s.rst'%basename
            outrstfile = os.path.join(rstdir, rstfile)

            # XXX: We might consider putting extra metadata in the example
            # files to include a title. If so, this line is where we would add
            # this information.
            fhsubdirIndex.write('    %s <%s>\n'%(os.path.basename(basename),rstfile))

            do_plot = (subdir in example_subdirs
                       and not noplot_regex.search(contents))
            if not do_plot:
                fhstatic = io.open(outputfile, 'w', encoding='utf-8')
                fhstatic.write(contents)
                fhstatic.close()

            if not out_of_date(fullpath, outrstfile):
                continue

            fh = io.open(outrstfile, 'w', encoding='utf-8')
            fh.write(u'.. _%s-%s:\n\n' % (subdir, basename))
            title = '%s example code: %s'%(subdir, fname)
            #title = '<img src=%s> %s example code: %s'%(thumbfile, subdir, fname)

            fh.write(title + u'\n')
            fh.write(u'=' * len(title) + u'\n\n')

            if do_plot:
                fh.write(u"\n\n.. plot:: %s\n\n::\n\n" % fullpath)
            else:
                fh.write(u"[`source code <%s>`_]\n\n::\n\n" % fname)

            # indent the contents
            contents = u'\n'.join([u'    %s'%row.rstrip() for row in contents.split(u'\n')])
            fh.write(contents)

            fh.write(u'\n\nKeywords: python, matplotlib, pylab, example, codex (see :ref:`how-to-search-examples`)')
            fh.close()

        fhsubdirIndex.close()

    fhindex.close()

    print()

def setup(app):
    app.connect('builder-inited', generate_example_rst)

    try: # multiple plugins may use mpl_example_sections
        app.add_config_value('mpl_example_sections', [], True)
    except sphinx.errors.ExtensionError:
        pass # mpl_example_sections already defined

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
