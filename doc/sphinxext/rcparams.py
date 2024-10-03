import enum

from docutils.parsers.rst import Directive

from matplotlib import rcParams, rcParamsDefault, rcsetup


def determine_type(validator):
    if isinstance(validator, enum.EnumType):
        return f'`~._enums.{validator.__name__}`'
    elif docstr := getattr(validator, '__doc__'):
        return docstr
    else:
        return str(validator)


def run(state_machine):
    lines = []
    current_section = None
    for rc in sorted(rcParams.keys()):
        if rc[0] == '_':
            continue

        # This would be much easier with #25617.
        section = rc.rsplit('.', maxsplit=1)[0]
        if section != current_section:
            lines.append(f'.. _rc-{section}:')
            current_section = section

        type_str = determine_type(rcsetup._validators[rc])
        if rc in rcParamsDefault and rc != "backend":
            default = f', default: {rcParamsDefault[rc]!r}'
        else:
            default = ''
        lines += [
            f'.. _rc-{rc}:',
            '',
            rc,
            '^' * len(rc),
            f'{type_str}{default}',
            f'    Documentation for {rc}.'
        ]
    state_machine.insert_input(lines, "rcParams table")
    return []


class RcParamsDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        return run(self.state_machine)


def setup(app):
    app.add_directive("rcparams", RcParamsDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
