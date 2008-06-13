"""A special directive for including a matplotlib plot.

Given a path to a .py file, it includes the source code inline, then:

- On HTML, will include a .png with a link to a high-res .png.
  Underneath that, a [source] link will go to a plain text .py of
  the source.

- On LaTeX, will include a .pdf
"""

from docutils.parsers.rst import directives
import os.path

try:
    # docutils 0.4
    from docutils.parsers.rst.directives.images import align
except ImportError:
    # docutils 0.5
    from docutils.parsers.rst.directives.images import Image
    align = Image.align

options = {'alt': directives.unchanged,
           'height': directives.length_or_unitless,
           'width': directives.length_or_percentage_or_unitless,
           'scale': directives.nonnegative_int,
           'align': align,
           'class': directives.class_option}

template = """
.. literalinclude:: %(reference)s.py

.. htmlonly::

   .. image:: %(reference)s.png
      :target: %(reference)s.hires.png
%(options)s

   `[original %(basename)s.py] <%(reference)s.py>`_

.. latexonly::
   .. image:: %(reference)s.pdf
%(options)s

"""

def run(arguments, options, state_machine, lineno):
    reference = directives.uri(arguments[0])
    if reference.endswith('.py'):
        reference = reference[:-3]
    options = ['      :%s: %s' % (key, val) for key, val in
               options.items()]
    options = "\n".join(options)
    basename = os.path.basename(reference)
    lines = template % locals()
    lines = lines.split('\n')

    state_machine.insert_input(
        lines, state_machine.input_lines.source(0))
    return []

try:
    from docutils.parsers.rst import Directive
except ImportError:
    from docutils.parsers.rst.directives import _directives

    def plot_directive(name, arguments, options, content, lineno,
                       content_offset, block_text, state, state_machine):
        return run(arguments, options, state_machine, lineno)
    plot_directive.__doc__ = __doc__
    plot_directive.arguments = (1, 0, 1)
    plot_directive.options = options

    _directives['plot'] = plot_directive
else:
    class plot_directive(Directive):
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = options
        def run(self):
            return run(self.arguments, self.options,
                       self.state_machine, self.lineno)
    plot_directive.__doc__ = __doc__

    directives.register_directive('plot', plot_directive)

