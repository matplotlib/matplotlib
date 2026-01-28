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
        for param in rcsetup._params:
            if param.name[0] == '_':
                continue
            lines += [
                f'.. _rcparam_{param.name.replace(".", "_")}:',
                '',
                f'{param.name}: ``{param.default!r}``',
                f'    {param.description if param.description else "*no description*"}'
            ]
        self.state_machine.insert_input(lines, 'rcParams table')
        return []


def setup(app):
    app.add_directive("rcparams", RcParamsDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
