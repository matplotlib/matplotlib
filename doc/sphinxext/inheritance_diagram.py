"""
Defines a docutils directive for inserting inheritance diagrams.

Provide the directive with one or more classes or modules (separated
by whitespace).  For modules, all of the classes in that module will
be used.

Example::

   Given the following classes:

   class A: pass
   class B(A): pass
   class C(A): pass
   class D(B, C): pass
   class E(B): pass

   .. inheritance-diagram: D E

   Produces a graph like the following:

               A
              / \
             B   C
            / \ /
           E   D

The graph is inserted as a PNG+image map into HTML and a PDF in
LaTeX.
"""

import inspect
import os
import re
import subprocess
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

from docutils.nodes import Body, Element
from docutils.parsers.rst import directives
from sphinx.roles import xfileref_role

def my_import(name):
    """Module importer - taken from the python documentation.

    This function allows importing names with dots in them."""
    
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class DotException(Exception):
    pass

class InheritanceGraph(object):
    """
    Given a list of classes, determines the set of classes that
    they inherit from all the way to the root "object", and then
    is able to generate a graphviz dot graph from them.
    """
    def __init__(self, class_names, show_builtins=False):
        """
        *class_names* is a list of child classes to show bases from.

        If *show_builtins* is True, then Python builtins will be shown
        in the graph.
        """
        self.class_names = class_names
        self.classes = self._import_classes(class_names)
        self.all_classes = self._all_classes(self.classes)
        if len(self.all_classes) == 0:
            raise ValueError("No classes found for inheritance diagram")
        self.show_builtins = show_builtins

    py_sig_re = re.compile(r'''^([\w.]*\.)?    # class names
                           (\w+)  \s* $        # optionally arguments
                           ''', re.VERBOSE)

    def _import_class_or_module(self, name):
        """
        Import a class using its fully-qualified *name*.
        """
        try:
            path, base = self.py_sig_re.match(name).groups()
        except:
            raise ValueError(
                "Invalid class or module '%s' specified for inheritance diagram" % name)
        fullname = (path or '') + base
        path = (path and path.rstrip('.'))
        if not path:
            path = base
        try:
            module = __import__(path, None, None, [])
            # We must do an import of the fully qualified name.  Otherwise if a
            # subpackage 'a.b' is requested where 'import a' does NOT provide
            # 'a.b' automatically, then 'a.b' will not be found below.  This
            # second call will force the equivalent of 'import a.b' to happen
            # after the top-level import above.
            my_import(fullname)
            
        except ImportError:
            raise ValueError(
                "Could not import class or module '%s' specified for inheritance diagram" % name)

        try:
            todoc = module
            for comp in fullname.split('.')[1:]:
                todoc = getattr(todoc, comp)
        except AttributeError:
            raise ValueError(
                "Could not find class or module '%s' specified for inheritance diagram" % name)

        # If a class, just return it
        if inspect.isclass(todoc):
            return [todoc]
        elif inspect.ismodule(todoc):
            classes = []
            for cls in todoc.__dict__.values():
                if inspect.isclass(cls) and cls.__module__ == todoc.__name__:
                    classes.append(cls)
            return classes
        raise ValueError(
            "'%s' does not resolve to a class or module" % name)

    def _import_classes(self, class_names):
        """
        Import a list of classes.
        """
        classes = []
        for name in class_names:
            classes.extend(self._import_class_or_module(name))
        return classes

    def _all_classes(self, classes):
        """
        Return a list of all classes that are ancestors of *classes*.
        """
        all_classes = {}

        def recurse(cls):
            all_classes[cls] = None
            for c in cls.__bases__:
                if c not in all_classes:
                    recurse(c)

        for cls in classes:
            recurse(cls)

        return all_classes.keys()

    def class_name(self, cls, parts=0):
        """
        Given a class object, return a fully-qualified name.  This
        works for things I've tested in matplotlib so far, but may not
        be completely general.
        """
        module = cls.__module__
        if module == '__builtin__':
            fullname = cls.__name__
        else:
            fullname = "%s.%s" % (module, cls.__name__)
        if parts == 0:
            return fullname
        name_parts = fullname.split('.')
        return '.'.join(name_parts[-parts:])

    def get_all_class_names(self):
        """
        Get all of the class names involved in the graph.
        """
        return [self.class_name(x) for x in self.all_classes]

    # These are the default options for graphviz
    default_graph_options = {
        "rankdir": "LR",
        "size": '"8.0, 12.0"'
        }
    default_node_options = {
        "shape": "box",
        "fontsize": 10,
        "height": 0.25,
        "fontname": "Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans",
        "style": '"setlinewidth(0.5)"'
        }
    default_edge_options = {
        "arrowsize": 0.5,
        "style": '"setlinewidth(0.5)"'
        }

    def _format_node_options(self, options):
        return ','.join(["%s=%s" % x for x in options.items()])
    def _format_graph_options(self, options):
        return ''.join(["%s=%s;\n" % x for x in options.items()])

    def generate_dot(self, fd, name, parts=0, urls={},
                     graph_options={}, node_options={},
                     edge_options={}):
        """
        Generate a graphviz dot graph from the classes that
        were passed in to __init__.

        *fd* is a Python file-like object to write to.

        *name* is the name of the graph

        *urls* is a dictionary mapping class names to http urls

        *graph_options*, *node_options*, *edge_options* are
        dictionaries containing key/value pairs to pass on as graphviz
        properties.
        """
        g_options = self.default_graph_options.copy()
        g_options.update(graph_options)
        n_options = self.default_node_options.copy()
        n_options.update(node_options)
        e_options = self.default_edge_options.copy()
        e_options.update(edge_options)

        fd.write('digraph %s {\n' % name)
        fd.write(self._format_graph_options(g_options))

        for cls in self.all_classes:
            if not self.show_builtins and cls in __builtins__.values():
                continue

            name = self.class_name(cls, parts)

            # Write the node
            this_node_options = n_options.copy()
            url = urls.get(self.class_name(cls))
            if url is not None:
                this_node_options['URL'] = '"%s"' % url
            fd.write('  "%s" [%s];\n' %
                     (name, self._format_node_options(this_node_options)))

            # Write the edges
            for base in cls.__bases__:
                if not self.show_builtins and base in __builtins__.values():
                    continue

                base_name = self.class_name(base, parts)
                fd.write('  "%s" -> "%s" [%s];\n' %
                         (base_name, name,
                          self._format_node_options(e_options)))
        fd.write('}\n')

    def run_dot(self, args, name, parts=0, urls={},
                graph_options={}, node_options={}, edge_options={}):
        """
        Run graphviz 'dot' over this graph, returning whatever 'dot'
        writes to stdout.

        *args* will be passed along as commandline arguments.

        *name* is the name of the graph

        *urls* is a dictionary mapping class names to http urls

        Raises DotException for any of the many os and
        installation-related errors that may occur.
        """
        try:
            dot = subprocess.Popen(['dot'] + list(args),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   close_fds=True)
        except OSError:
            raise DotException("Could not execute 'dot'.  Are you sure you have 'graphviz' installed?")
        except ValueError:
            raise DotException("'dot' called with invalid arguments")
        except:
            raise DotException("Unexpected error calling 'dot'")

        self.generate_dot(dot.stdin, name, parts, urls, graph_options,
                          node_options, edge_options)
        dot.stdin.close()
        result = dot.stdout.read()
        returncode = dot.wait()
        if returncode != 0:
            raise DotException("'dot' returned the errorcode %d" % returncode)
        return result

class inheritance_diagram(Body, Element):
    """
    A docutils node to use as a placeholder for the inheritance
    diagram.
    """
    pass

def inheritance_diagram_directive(name, arguments, options, content, lineno,
                                  content_offset, block_text, state,
                                  state_machine):
    """
    Run when the inheritance_diagram directive is first encountered.
    """
    node = inheritance_diagram()

    class_names = arguments

    # Create a graph starting with the list of classes
    graph = InheritanceGraph(class_names)

    # Create xref nodes for each target of the graph's image map and
    # add them to the doc tree so that Sphinx can resolve the
    # references to real URLs later.  These nodes will eventually be
    # removed from the doctree after we're done with them.
    for name in graph.get_all_class_names():
        refnodes, x = xfileref_role(
            'class', ':class:`%s`' % name, name, 0, state)
        node.extend(refnodes)
    # Store the graph object so we can use it to generate the
    # dot file later
    node['graph'] = graph
    # Store the original content for use as a hash
    node['parts'] = options.get('parts', 0)
    node['content'] = " ".join(class_names)
    return [node]

def get_graph_hash(node):
    return md5(node['content'] + str(node['parts'])).hexdigest()[-10:]

def html_output_graph(self, node):
    """
    Output the graph for HTML.  This will insert a PNG with clickable
    image map.
    """
    graph = node['graph']
    parts = node['parts']

    graph_hash = get_graph_hash(node)
    name = "inheritance%s" % graph_hash
    path = '_images'
    dest_path = os.path.join(setup.app.builder.outdir, path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    png_path = os.path.join(dest_path, name + ".png")
    path = setup.app.builder.imgpath

    # Create a mapping from fully-qualified class names to URLs.
    urls = {}
    for child in node:
        if child.get('refuri') is not None:
            urls[child['reftitle']] = child.get('refuri')
        elif child.get('refid') is not None:
            urls[child['reftitle']] = '#' + child.get('refid')

    # These arguments to dot will save a PNG file to disk and write
    # an HTML image map to stdout.
    image_map = graph.run_dot(['-Tpng', '-o%s' % png_path, '-Tcmapx'],
                              name, parts, urls)
    return ('<img src="%s/%s.png" usemap="#%s" class="inheritance"/>%s' %
            (path, name, name, image_map))

def latex_output_graph(self, node):
    """
    Output the graph for LaTeX.  This will insert a PDF.
    """
    graph = node['graph']
    parts = node['parts']

    graph_hash = get_graph_hash(node)
    name = "inheritance%s" % graph_hash
    dest_path = os.path.abspath(os.path.join(setup.app.builder.outdir, '_images'))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    pdf_path = os.path.abspath(os.path.join(dest_path, name + ".pdf"))

    graph.run_dot(['-Tpdf', '-o%s' % pdf_path],
                  name, parts, graph_options={'size': '"6.0,6.0"'})
    return '\n\\includegraphics{%s}\n\n' % pdf_path

def visit_inheritance_diagram(inner_func):
    """
    This is just a wrapper around html/latex_output_graph to make it
    easier to handle errors and insert warnings.
    """
    def visitor(self, node):
        try:
            content = inner_func(self, node)
        except DotException, e:
            # Insert the exception as a warning in the document
            warning = self.document.reporter.warning(str(e), line=node.line)
            warning.parent = node
            node.children = [warning]
        else:
            source = self.document.attributes['source']
            self.body.append(content)
            node.children = []
    return visitor

def do_nothing(self, node):
    pass

def setup(app):
    setup.app = app
    setup.confdir = app.confdir

    app.add_node(
        inheritance_diagram,
        latex=(visit_inheritance_diagram(latex_output_graph), do_nothing),
        html=(visit_inheritance_diagram(html_output_graph), do_nothing))
    app.add_directive(
        'inheritance-diagram', inheritance_diagram_directive,
        False, (1, 100, 0), parts = directives.nonnegative_int)
