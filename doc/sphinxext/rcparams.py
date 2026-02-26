from docutils.parsers.rst import Directive

from matplotlib import rcsetup


class RcParamsDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        """
        Generate rst documentation for rcParams.

        Note: The style is very simple, but will be refined later.
        """
        self.state.document.settings.env.note_dependency(__file__)
        self.state.document.settings.env.note_dependency(rcsetup.__file__)
        lines = []
        for elem in rcsetup._DEFINITION:
            if isinstance(elem, (rcsetup._Section, rcsetup._Subsection)):
                title_char = '-' if isinstance(elem, rcsetup._Section) else '~'
                lines += [
                    '',
                    elem.title,
                    title_char * len(elem.title),
                    '',
                    elem.description or "",
                    '',
                ]
            elif isinstance(elem, rcsetup._Param):
                if elem.name[0] == '_':
                    continue
                lines += [
                    f'.. _rcparam_{elem.name.replace(".", "_")}:',
                    '',
                    f'{elem.name}: ``{elem.default!r}``',
                    f'   {elem.description if elem.description else "*no description*"}'
                ]
        self.state_machine.insert_input(lines, 'rcParams table')
        return []


def setup(app):
    app.add_directive("rcparams", RcParamsDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
