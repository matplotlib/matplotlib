"""Implementation of pydocstyle integration with Flake8.

pydocstyle docstrings convention needs error code and class parser for be
included as module into flake8
"""
import re

supports_ignore_inline_noqa = False
supports_property_decorators = False
supports_ignore_self_only_init = False
try:
    import pydocstyle as pep257

    module_name = "pydocstyle"

    pydocstyle_version = tuple(
        int(num) for num in pep257.__version__.split(".")
    )
    supports_ignore_inline_noqa = pydocstyle_version >= (6, 0, 0)
    supports_property_decorators = pydocstyle_version >= (6, 2, 0)
    supports_ignore_self_only_init = pydocstyle_version >= (6, 3, 0)
except ImportError:
    import pep257

    module_name = "pep257"

__version__ = "1.7.0"
__all__ = ("pep257Checker",)


class _ContainsAll:
    def __contains__(self, code):  # type: (str) -> bool
        return True


class EnvironError(pep257.Error):
    def __init__(self, err):
        super().__init__(
            code="D998",
            short_desc="EnvironmentError: " + str(err),
            context=None,
        )

    @property
    def line(self):
        """Return 0 as line number for EnvironmentError."""
        return 0


class AllError(pep257.Error):
    def __init__(self, err):
        super().__init__(
            code="D999",
            short_desc=str(err).partition("\n")[0],
            context=None,
        )

    @property
    def line(self):
        """pep257.AllError does not contain line number. Return 0 instead."""
        return 0


class pep257Checker:
    """Flake8 needs a class to check python file."""

    name = "flake8-docstrings"
    version = f"{__version__}, {module_name}: {pep257.__version__}"

    def __init__(self, tree, filename, lines):
        """Initialize the checker."""
        self.tree = tree
        self.filename = filename
        self.checker = pep257.ConventionChecker()
        self.source = "".join(lines)

    @classmethod
    def add_options(cls, parser):
        """Add plugin configuration option to flake8."""
        parser.add_option(
            "--docstring-convention",
            action="store",
            parse_from_config=True,
            default="pep257",
            choices=sorted(pep257.conventions) + ["all"],
            help=(
                "pydocstyle docstring convention, default 'pep257'. "
                "Use the special value 'all' to enable all codes (note: "
                "some codes are conflicting so you'll need to then exclude "
                "those)."
            ),
        )
        parser.add_option(
            "--ignore-decorators",
            action="store",
            parse_from_config=True,
            default=None,
            help=(
                "pydocstyle ignore-decorators regular expression, "
                "default None. "
                "Ignore any functions or methods that are decorated by "
                "a function with a name fitting this regular expression. "
                "The default is not ignore any decorated functions. "
            ),
        )

        if supports_property_decorators:
            from pydocstyle.config import ConfigurationParser

            default_property_decorators = (
                ConfigurationParser.DEFAULT_PROPERTY_DECORATORS
            )
            parser.add_option(
                "--property-decorators",
                action="store",
                parse_from_config=True,
                default=default_property_decorators,
                help=(
                    "consider any method decorated with one of these "
                    "decorators as a property, and consequently allow "
                    "a docstring which is not in imperative mood; default "
                    f"is --property-decorators='{default_property_decorators}'"
                ),
            )

        if supports_ignore_self_only_init:
            parser.add_option(
                "--ignore-self-only-init",
                action="store_true",
                parse_from_config=True,
                help="ignore __init__ methods which only have a self param.",
            )

    @classmethod
    def parse_options(cls, options):
        """Parse the configuration options given to flake8."""
        cls.convention = options.docstring_convention
        cls.ignore_decorators = (
            re.compile(options.ignore_decorators)
            if options.ignore_decorators
            else None
        )
        if supports_property_decorators:
            cls.property_decorators = options.property_decorators
        if supports_ignore_self_only_init:
            cls.ignore_self_only_init = options.ignore_self_only_init

    def _call_check_source(self):
        check_source_kwargs = {}
        if supports_ignore_inline_noqa:
            check_source_kwargs["ignore_inline_noqa"] = True
        if supports_property_decorators:
            check_source_kwargs["property_decorators"] = (
                set(self.property_decorators.split(","))
                if self.property_decorators
                else None
            )
        if supports_ignore_self_only_init:
            check_source_kwargs[
                "ignore_self_only_init"
            ] = self.ignore_self_only_init

        return self.checker.check_source(
            self.source,
            self.filename,
            ignore_decorators=self.ignore_decorators,
            **check_source_kwargs,
        )

    def _check_source(self):
        try:
            for err in self._call_check_source():
                yield err
        except pep257.AllError as err:
            yield AllError(err)
        except OSError as err:
            yield EnvironError(err)

    def run(self):
        """Use directly check() api from pydocstyle."""
        if self.convention == "all":
            checked_codes = _ContainsAll()
        else:
            checked_codes = pep257.conventions[self.convention] | {
                "D998",
                "D999",
            }
        for error in self._check_source():
            if isinstance(error, pep257.Error) and error.code in checked_codes:
                # NOTE(sigmavirus24): Fixes GitLab#3
                message = f"{error.code} {error.short_desc}"
                yield (error.line, 0, message, type(self))
