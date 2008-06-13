from pygments.lexer import Lexer, do_insertions
from pygments.lexers.agile import PythonConsoleLexer, PythonLexer, \
    PythonTracebackLexer
from pygments.token import Comment, Generic
from sphinx import highlighting
import re

line_re = re.compile('.*?\n')

class IPythonConsoleLexer(Lexer):
    """
    For IPython console output or doctests, such as:

    Tracebacks are not currently supported.

    .. sourcecode:: ipython

      In [1]: a = 'foo'

      In [2]: a
      Out[2]: 'foo'

      In [3]: print a
      foo

      In [4]: 1 / 0
    """
    name = 'IPython console session'
    aliases = ['ipython']
    mimetypes = ['text/x-ipython-console']
    input_prompt = re.compile("(In \[[0-9]+\]: )|(   \.\.\.+:)")
    output_prompt = re.compile("(Out\[[0-9]+\]: )|(   \.\.\.+:)")
    continue_prompt = re.compile("   \.\.\.+:")
    tb_start = re.compile("\-+")

    def get_tokens_unprocessed(self, text):
        pylexer = PythonLexer(**self.options)
        tblexer = PythonTracebackLexer(**self.options)

        curcode = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            input_prompt = self.input_prompt.match(line)
            continue_prompt = self.continue_prompt.match(line.rstrip())
            output_prompt = self.output_prompt.match(line)
            if line.startswith("#"):
                insertions.append((len(curcode),
                                   [(0, Comment, line)]))
            elif input_prompt is not None:
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt, input_prompt.group())]))
                curcode += line[input_prompt.end():]
            elif continue_prompt is not None:
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt, continue_prompt.group())]))
                curcode += line[continue_prompt.end():]
            elif output_prompt is not None:
                insertions.append((len(curcode),
                                   [(0, Generic.Output, output_prompt.group())]))
                curcode += line[output_prompt.end():]
            else:
                if curcode:
                    for item in do_insertions(insertions,
                                              pylexer.get_tokens_unprocessed(curcode)):
                        yield item
                        curcode = ''
                        insertions = []
                yield match.start(), Generic.Output, line
        if curcode:
            for item in do_insertions(insertions,
                                      pylexer.get_tokens_unprocessed(curcode)):
                yield item

highlighting.lexers['ipython'] = IPythonConsoleLexer()
