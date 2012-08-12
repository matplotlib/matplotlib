"""
generate the rst files for the examples by iterating over the pylab examples
"""
from __future__ import print_function
import os, glob

import os
import re
import sys
import token
import tokenize
fileList = []

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.

    TODO: this check isn't adequate in some cases.  Eg, if we discover
    a bug when building the examples, the original and derived will be
    unchanged but we still want to force a rebuild.
    """
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)

noplot_regex = re.compile(r"#\s*-\*-\s*noplot\s*-\*-")

def extract_docstring(lines):
    """ Extract a module-level docstring, if any
    """
    start_row = 0
    if lines[0].startswith('#!'):
        lines.pop(0)
        start_row = 1

    docstring = ''
    first_par = ''
    tokens = tokenize.generate_tokens(lines.__iter__().next)
    for tok_type, tok_content, _, (erow, _), _ in tokens:
        tok_type = token.tok_name[tok_type]
        if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
            continue
        elif tok_type == 'STRING':
            docstring = eval(tok_content)
            # If the docstring is formatted with several paragraphs, extract
            # the first one:
            paragraphs = '\n'.join(line.rstrip()
                              for line in docstring.split('\n')).split('\n\n')
            if len(paragraphs) > 0:
                first_par = paragraphs[0]
        break
    return docstring, first_par, erow + 1 + start_row



def generate_example_rst(app):
    rootdir = os.path.join(app.builder.srcdir, 'mpl_examples')
    exampledir = os.path.join(app.builder.srcdir, 'examples')
    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    datad = {}
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if ( fname.startswith('.') or fname.startswith('#')
                 or fname.startswith('_') or not fname.endswith('.py') ):
                continue

            fullpath = os.path.join(root,fname)
            contents = file(fullpath).readlines()
            # indent
            relpath = os.path.split(root)[-1]
            datad.setdefault(relpath, []).append((fullpath, fname, contents))

    subdirs = datad.keys()
    subdirs.sort()

    fhindex = file(os.path.join(exampledir, 'index.rst'), 'w')
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
        fhsubdirIndex = file(subdirIndexFile, 'w')
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

            fhsubdirIndex.write('    %s <%s>\n'%(os.path.basename(basename),rstfile))

            do_plot = (subdir in ('api',
                                  'pylab_examples',
                                  'units',
                                  'mplot3d',
                                  'axes_grid',
                                  ) and
                       not noplot_regex.search('\n'.join(contents)))
            if not do_plot:
                fhstatic = file(outputfile, 'w')
                fhstatic.write('\n'.join(contents))
                fhstatic.close()

            if not out_of_date(fullpath, outrstfile):
                continue

            docstring, short_desc, end_row = extract_docstring(contents)

            fh = file(outrstfile, 'w')
            fh.write('.. _%s-%s:\n\n'%(subdir, basename))
            title = '%s example code: %s'%(subdir, fname)
            #title = '<img src=%s> %s example code: %s'%(thumbfile, subdir, fname)

            fh.write(title + '\n')
            fh.write('='*len(title) + '\n\n')
            fh.write(docstring + '\n')

            if do_plot:
                fh.write("\n\n.. plot:: %s\n\n::\n\n" % fullpath)
            else:
                fh.write("[`source code <%s>`_]\n\n::\n\n" % fname)

            # indent the contents
            # Strip out the docstring and short_desc portions
            contents = '\n'.join(['    %s'%row.rstrip() for
                                  row in contents[end_row:]])
            fh.write(contents)

            fh.write('\n\nKeywords: python, matplotlib, pylab, example, codex (see :ref:`how-to-search-examples`)')
            fh.close()

        fhsubdirIndex.close()

    fhindex.close()

    print()

def setup(app):
    app.connect('builder-inited', generate_example_rst)
