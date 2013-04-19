"""
A directive for including a matplotlib plot in a Sphinx document.

By default, in HTML output, `plot` will include a .png file with a
link to a high-res .png and .pdf.  In LaTeX output, it will include a
.pdf.

The source code for the plot may be included in one of three ways:

  1. **A path to a source file** as the argument to the directive::

       .. plot:: path/to/plot.py

     When a path to a source file is given, the content of the
     directive may optionally contain a caption for the plot::

       .. plot:: path/to/plot.py

          This is the caption for the plot

     Additionally, one my specify the name of a function to call (with
     no arguments) immediately after importing the module::

       .. plot:: path/to/plot.py plot_function1

  2. Included as **inline content** to the directive::

       .. plot::

          import matplotlib.pyplot as plt
          import matplotlib.image as mpimg
          import numpy as np
          img = mpimg.imread('_static/stinkbug.png')
          imgplot = plt.imshow(img)

  3. Using **doctest** syntax::

       .. plot::
          A plotting example:
          >>> import matplotlib.pyplot as plt
          >>> plt.plot([1,2,3], [4,5,6])

Options
-------

The ``plot`` directive supports the following options:

    format : {'python', 'doctest'}
        Specify the format of the input

    include-source : bool
        Whether to display the source code. The default can be changed
        using the `plot_include_source` variable in conf.py

    encoding : str
        If this source file is in a non-UTF8 or non-ASCII encoding,
        the encoding must be specified using the `:encoding:` option.
        The encoding will not be inferred using the ``-*- coding -*-``
        metacomment.

    context : bool
        If provided, the code will be run in the context of all
        previous plot directives for which the `:context:` option was
        specified.  This only applies to inline code plot directives,
        not those run from files.

    nofigs : bool
        If specified, the code block will be run, but no figures will
        be inserted.  This is usually useful with the ``:context:``
        option.

Additionally, this directive supports all of the options of the
`image` directive, except for `target` (since plot will add its own
target).  These include `alt`, `height`, `width`, `scale`, `align` and
`class`.

Configuration options
---------------------

The plot directive has the following configuration options:

    plot_include_source
        Default value for the include-source option

    plot_pre_code
        Code that should be executed before each plot.

    plot_basedir
        Base directory, to which ``plot::`` file names are relative
        to.  (If None or empty, file names are relative to the
        directoly where the file containing the directive is.)

    plot_formats
        File formats to generate. List of tuples or strings::

            [(suffix, dpi), suffix, ...]

        that determine the file format and the DPI. For entries whose
        DPI was omitted, sensible defaults are chosen.

    plot_html_show_formats
        Whether to show links to the files in HTML.

    plot_rcparams
        A dictionary containing any non-standard rcParams that should
        be applied before each plot.

    plot_apply_rcparams
        By default, rcParams are applied when `context` option is not used in 
        a plot  directive.  This configuration option overrides this behaviour 
        and applies rcParams before each plot.

    plot_working_directory
        By default, the working directory will be changed to the directory of 
        the example, so the code can get at its data files, if any.  Also its 
        path will be added to `sys.path` so it can import any helper modules 
        sitting beside it.  This configuration option can be used to specify 
        a central directory (also added to `sys.path`) where data files and 
        helper modules for all code are located. 

    plot_template
        Provide a customized template for preparing resturctured text.
        

"""
from __future__ import print_function

import sys, os, glob, shutil, imp, warnings, cStringIO, re, textwrap
import traceback

from docutils.parsers.rst import directives
from docutils import nodes
from docutils.parsers.rst.directives.images import Image
align = Image.align
import sphinx

sphinx_version = sphinx.__version__.split(".")
# The split is necessary for sphinx beta versions where the string is
# '6b1'
sphinx_version = tuple([int(re.split('[a-z]', x)[0])
                        for x in sphinx_version[:2]])

try:
    # Sphinx depends on either Jinja or Jinja2
    import jinja2
    def format_template(template, **kw):
        return jinja2.Template(template).render(**kw)
except ImportError:
    import jinja
    def format_template(template, **kw):
        return jinja.from_string(template, **kw)

import matplotlib
import matplotlib.cbook as cbook
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

__version__ = 2

#------------------------------------------------------------------------------
# Relative pathnames
#------------------------------------------------------------------------------

# os.path.relpath is new in Python 2.6
try:
    from os.path import relpath
except ImportError:
    # Copied from Python 2.7
    if 'posix' in sys.builtin_module_names:
        def relpath(path, start=os.path.curdir):
            """Return a relative version of a path"""
            from os.path import sep, curdir, join, abspath, commonprefix, \
                 pardir

            if not path:
                raise ValueError("no path specified")

            start_list = abspath(start).split(sep)
            path_list = abspath(path).split(sep)

            # Work out how much of the filepath is shared by start and path.
            i = len(commonprefix([start_list, path_list]))

            rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
            if not rel_list:
                return curdir
            return join(*rel_list)
    elif 'nt' in sys.builtin_module_names:
        def relpath(path, start=os.path.curdir):
            """Return a relative version of a path"""
            from os.path import sep, curdir, join, abspath, commonprefix, \
                 pardir, splitunc

            if not path:
                raise ValueError("no path specified")
            start_list = abspath(start).split(sep)
            path_list = abspath(path).split(sep)
            if start_list[0].lower() != path_list[0].lower():
                unc_path, rest = splitunc(path)
                unc_start, rest = splitunc(start)
                if bool(unc_path) ^ bool(unc_start):
                    raise ValueError("Cannot mix UNC and non-UNC paths (%s and %s)"
                                                                        % (path, start))
                else:
                    raise ValueError("path is on drive %s, start on drive %s"
                                                        % (path_list[0], start_list[0]))
            # Work out how much of the filepath is shared by start and path.
            for i in range(min(len(start_list), len(path_list))):
                if start_list[i].lower() != path_list[i].lower():
                    break
            else:
                i += 1

            rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
            if not rel_list:
                return curdir
            return join(*rel_list)
    else:
        raise RuntimeError("Unsupported platform (no relpath available!)")

#------------------------------------------------------------------------------
# Registration hook
#------------------------------------------------------------------------------

def plot_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):
    return run(arguments, content, options, state_machine, state, lineno)
plot_directive.__doc__ = __doc__

def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:
        raise ValueError('"%s" unknown boolean' % arg)

def _option_format(arg):
    return directives.choice(arg, ('python', 'doctest'))

def _option_align(arg):
    return directives.choice(arg, ("top", "middle", "bottom", "left", "center",
                                   "right"))

def mark_plot_labels(app, document):
    """
    To make plots referenceable, we need to move the reference from
    the "htmlonly" (or "latexonly") node to the actual figure node
    itself.
    """
    for name, explicit in document.nametypes.iteritems():
        if not explicit:
            continue
        labelid = document.nameids[name]
        if labelid is None:
            continue
        node = document.ids[labelid]
        if node.tagname in ('html_only', 'latex_only'):
            for n in node:
                if n.tagname == 'figure':
                    sectname = name
                    for c in n:
                        if c.tagname == 'caption':
                            sectname = c.astext()
                            break

                    node['ids'].remove(labelid)
                    node['names'].remove(name)
                    n['ids'].append(labelid)
                    n['names'].append(name)
                    document.settings.env.labels[name] = \
                        document.settings.env.docname, labelid, sectname
                    break

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    options = {'alt': directives.unchanged,
               'height': directives.length_or_unitless,
               'width': directives.length_or_percentage_or_unitless,
               'scale': directives.nonnegative_int,
               'align': _option_align,
               'class': directives.class_option,
               'include-source': _option_boolean,
               'format': _option_format,
               'context': directives.flag,
               'nofigs': directives.flag,
               'encoding': directives.encoding
               }

    app.add_directive('plot', plot_directive, True, (0, 2, False), **options)
    app.add_config_value('plot_pre_code', None, True)
    app.add_config_value('plot_include_source', False, True)
    app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)
    app.add_config_value('plot_basedir', None, True)
    app.add_config_value('plot_html_show_formats', True, True)
    app.add_config_value('plot_rcparams', {}, True)
    app.add_config_value('plot_apply_rcparams', False, True)
    app.add_config_value('plot_working_directory', None, True)
    app.add_config_value('plot_template', None, True)

    app.connect('doctree-read', mark_plot_labels)

#------------------------------------------------------------------------------
# Doctest handling
#------------------------------------------------------------------------------

def contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, '<string>', 'exec')
        return False
    except SyntaxError:
        pass
    r = re.compile(r'^\s*>>>', re.M)
    m = r.search(text)
    return bool(m)

def unescape_doctest(text):
    """
    Extract code from a piece of text, which contains either Python code
    or doctests.

    """
    if not contains_doctest(text):
        return text

    code = ""
    for line in text.split("\n"):
        m = re.match(r'^\s*(>>>|\.\.\.) (.*)$', line)
        if m:
            code += m.group(2) + "\n"
        elif line.strip():
            code += "# " + line.strip() + "\n"
        else:
            code += "\n"
    return code

def split_code_at_show(text):
    """
    Split code at plt.show()

    """

    parts = []
    is_doctest = contains_doctest(text)

    part = []
    for line in text.split("\n"):
        if (not is_doctest and line.strip() == 'plt.show()') or \
               (is_doctest and line.strip() == '>>> plt.show()'):
            part.append(line)
            parts.append("\n".join(part))
            part = []
        else:
            part.append(line)
    if "\n".join(part).strip():
        parts.append("\n".join(part))
    return parts

#------------------------------------------------------------------------------
# Template
#------------------------------------------------------------------------------


TEMPLATE = """
{{ source_code }}

{{ only_html }}

   {% if source_link or (html_show_formats and not multi_image) %}
   (
   {%- if source_link -%}
   `Source code <{{ source_link }}>`__
   {%- endif -%}
   {%- if html_show_formats and not multi_image -%}
     {%- for img in images -%}
       {%- for fmt in img.formats -%}
         {%- if source_link or not loop.first -%}, {% endif -%}
         `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
       {%- endfor -%}
     {%- endfor -%}
   {%- endif -%}
   )
   {% endif %}

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.png
      {%- for option in options %}
      {{ option }}
      {% endfor %}

      {% if html_show_formats and multi_image -%}
        (
        {%- for fmt in img.formats -%}
        {%- if not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
        {%- endfor -%}
        )
      {%- endif -%}

      {{ caption }}
   {% endfor %}

{{ only_latex }}

   {% for img in images %}
   .. image:: {{ build_dir }}/{{ img.basename }}.pdf
   {% endfor %}

{{ only_texinfo }}

   {% for img in images %}
   .. image:: {{ build_dir }}/{{ img.basename }}.png
      {%- for option in options %}
      {{ option }}
      {% endfor %}

   {% endfor %}

"""

exception_template = """
.. htmlonly::

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# the context of the plot for all directives specified with the
# :context: option
plot_context = dict()

class ImageFile(object):
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, "%s.%s" % (self.basename, format))

    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return (not os.path.exists(derived) or
            (os.path.exists(original) and
             os.stat(derived).st_mtime < os.stat(original).st_mtime))

class PlotError(RuntimeError):
    pass

def run_code(code, code_path, ns=None, function_name=None):
    """
    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """

    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.  Add its path to sys.path
    # so it can import any helper modules sitting beside it.

    pwd = os.getcwd()
    old_sys_path = list(sys.path)
    if setup.config.plot_working_directory is not None:
        try:
            os.chdir(setup.config.plot_working_directory)
        except OSError as err:
            raise OSError(str(err) + '\n`plot_working_directory` option in'
                          'Sphinx configuration file must be a valid '
                          'directory path')
        except TypeError as err:
            raise TypeError(str(err) + '\n`plot_working_directory` option in '
                            'Sphinx configuration file must be a string or '
                            'None')
        sys.path.insert(0, setup.config.plot_working_directory)
    elif code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)
        sys.path.insert(0, dirname)

    # Redirect stdout
    stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()

    # Reset sys.argv
    old_sys_argv = sys.argv
    sys.argv = [code_path]

    try:
        try:
            code = unescape_doctest(code)
            if ns is None:
                ns = {}
            if not ns:
                if setup.config.plot_pre_code is None:
                    exec "import numpy as np\nfrom matplotlib import pyplot as plt\n" in ns
                else:
                    exec setup.config.plot_pre_code in ns
            if "__main__" in code:
                exec "__name__ = '__main__'" in ns
            exec code in ns
            if function_name is not None:
                exec function_name + "()" in ns
        except (Exception, SystemExit), err:
            raise PlotError(traceback.format_exc())
    finally:
        os.chdir(pwd)
        sys.argv = old_sys_argv
        sys.path[:] = old_sys_path
        sys.stdout = stdout
    return ns

def clear_state(plot_rcparams):
    plt.close('all')
    matplotlib.rc_file_defaults()
    matplotlib.rcParams.update(plot_rcparams)

def render_figures(code, code_path, output_dir, output_base, context,
                   function_name, config):
    """
    Run a pyplot script and save the low and high res PNGs and a PDF
    in outdir.

    Save the images under *output_dir* with file names derived from
    *output_base*
    """
    # -- Parse format list
    default_dpi = {'png': 80, 'hires.png': 200, 'pdf': 200}
    formats = []
    plot_formats = config.plot_formats
    if isinstance(plot_formats, (str, unicode)):
        plot_formats = eval(plot_formats)
    for fmt in plot_formats:
        if isinstance(fmt, str):
            formats.append((fmt, default_dpi.get(fmt, 80)))
        elif type(fmt) in (tuple, list) and len(fmt)==2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise PlotError('invalid image format "%r" in plot_formats' % fmt)

    # -- Try to determine if all images already exist

    code_pieces = split_code_at_show(code)

    # Look for single-figure output files first
    # Look for single-figure output files first
    all_exists = True
    img = ImageFile(output_base, output_dir)
    for format, dpi in formats:
        if out_of_date(code_path, img.filename(format)):
            all_exists = False
            break
        img.formats.append(format)

    if all_exists:
        return [(code, [img])]

    # Then look for multi-figure output files
    results = []
    all_exists = True
    for i, code_piece in enumerate(code_pieces):
        images = []
        for j in xrange(1000):
            if len(code_pieces) > 1:
                img = ImageFile('%s_%02d_%02d' % (output_base, i, j), output_dir)
            else:
                img = ImageFile('%s_%02d' % (output_base, j), output_dir)
            for format, dpi in formats:
                if out_of_date(code_path, img.filename(format)):
                    all_exists = False
                    break
                img.formats.append(format)

            # assume that if we have one, we have them all
            if not all_exists:
                all_exists = (j > 0)
                break
            images.append(img)
        if not all_exists:
            break
        results.append((code_piece, images))

    if all_exists:
        return results

    # We didn't find the files, so build them

    results = []
    if context:
        ns = plot_context
    else:
        ns = {}

    for i, code_piece in enumerate(code_pieces):
        if not context or config.plot_apply_rcparams:
            clear_state(config.plot_rcparams)
        run_code(code_piece, code_path, ns, function_name)

        images = []
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        for j, figman in enumerate(fig_managers):
            if len(fig_managers) == 1 and len(code_pieces) == 1:
                img = ImageFile(output_base, output_dir)
            elif len(code_pieces) == 1:
                img = ImageFile("%s_%02d" % (output_base, j), output_dir)
            else:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j),
                                output_dir)
            images.append(img)
            for format, dpi in formats:
                try:
                    figman.canvas.figure.savefig(img.filename(format), dpi=dpi)
                except Exception,err:
                    raise PlotError(traceback.format_exc())
                img.formats.append(format)

        results.append((code_piece, images))

    if not context or config.plot_apply_rcparams:
        clear_state(config.plot_rcparams)

    return results

def run(arguments, content, options, state_machine, state, lineno):
    # The user may provide a filename *or* Python code content, but not both
    if arguments and content:
        raise RuntimeError("plot:: directive can't have both args and content")

    document = state_machine.document
    config = document.settings.env.config
    nofigs = options.has_key('nofigs')

    options.setdefault('include-source', config.plot_include_source)
    context = options.has_key('context')

    rst_file = document.attributes['source']
    rst_dir = os.path.dirname(rst_file)

    if len(arguments):
        if not config.plot_basedir:
            source_file_name = os.path.join(setup.app.builder.srcdir,
                                            directives.uri(arguments[0]))
        else:
            source_file_name = os.path.join(setup.confdir, config.plot_basedir,
                                            directives.uri(arguments[0]))

        # If there is content, it will be passed as a caption.
        caption = '\n'.join(content)

        # If the optional function name is provided, use it
        if len(arguments) == 2:
            function_name = arguments[1]
        else:
            function_name = None

        with open(source_file_name, 'r') as fd:
            code = fd.read()
        output_base = os.path.basename(source_file_name)
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        output_base = '%s-%d.py' % (base, counter)
        function_name = None
        caption = ''

    base, source_ext = os.path.splitext(output_base)
    if source_ext in ('.py', '.rst', '.txt'):
        output_base = base
    else:
        source_ext = ''

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace('.', '-')

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if options.has_key('format'):
        if options['format'] == 'python':
            is_doctest = False
        else:
            is_doctest = True

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = os.path.dirname(source_rel_name)
    while source_rel_dir.startswith(os.path.sep):
        source_rel_dir = source_rel_dir[1:]

    # build_dir: where to place output files (temporarily)
    build_dir = os.path.join(os.path.dirname(setup.app.doctreedir),
                             'plot_directive',
                             source_rel_dir)
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # output_dir: final location in the builder's directory
    dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
                                            source_rel_dir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir) # no problem here for me, but just use built-ins

    # how to link to files from the RST file
    dest_dir_link = os.path.join(relpath(setup.confdir, rst_dir),
                                 source_rel_dir).replace(os.path.sep, '/')
    build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
    source_link = dest_dir_link + '/' + output_base + source_ext

    # make figures
    try:
        results = render_figures(code, source_file_name, build_dir, output_base,
                                 context, function_name, config)
        errors = []
    except PlotError, err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2, "Exception occurred in plotting %s\n from %s:\n%s" % (output_base,
                                                source_file_name, err),
            line=lineno)
        results = [(code, [])]
        errors = [sm]

    # Properly indent the caption
    caption = '\n'.join('      ' + line.strip()
                        for line in caption.split('\n'))

    # generate output restructuredtext
    total_lines = []
    for j, (code_piece, images) in enumerate(results):
        if options['include-source']:
            if is_doctest:
                lines = ['']
                lines += [row.rstrip() for row in code_piece.split('\n')]
            else:
                lines = ['.. code-block:: python', '']
                lines += ['    %s' % row.rstrip()
                          for row in code_piece.split('\n')]
            source_code = "\n".join(lines)
        else:
            source_code = ""

        if nofigs:
            images = []

        opts = [':%s: %s' % (key, val) for key, val in options.items()
                if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]

        only_html = ".. only:: html"
        only_latex = ".. only:: latex"
        only_texinfo = ".. only:: texinfo"

        if j == 0:
            src_link = source_link
        else:
            src_link = None

        result = format_template(
            config.plot_template or TEMPLATE,
            dest_dir=dest_dir_link,
            build_dir=build_dir_link,
            source_link=src_link,
            multi_image=len(images) > 1,
            only_html=only_html,
            only_latex=only_latex,
            only_texinfo=only_texinfo,
            options=opts,
            images=images,
            source_code=source_code,
            html_show_formats=config.plot_html_show_formats,
            caption=caption)

        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # copy image files to builder's output directory, if necessary
    if not os.path.exists(dest_dir):
        cbook.mkdirs(dest_dir)

    for code_piece, images in results:
        for img in images:
            for fn in img.filenames():
                destimg = os.path.join(dest_dir, os.path.basename(fn))
                if fn != destimg:
                    shutil.copyfile(fn, destimg)

    # copy script (if necessary)
    target_name = os.path.join(dest_dir, output_base + source_ext)
    with open(target_name, 'w') as f:
        if source_file_name == rst_file:
            code_escaped = unescape_doctest(code)
        else:
            code_escaped = code
        f.write(code_escaped)

    return errors
