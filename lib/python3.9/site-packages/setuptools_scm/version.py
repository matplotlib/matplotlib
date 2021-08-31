import datetime
import warnings
import re
import time
import os

from .config import Configuration
from .utils import trace

try:
    from packaging.version import Version
except ImportError:
    import pkg_resources

    Version = pkg_resources.packaging.version.Version


from pkg_resources import iter_entry_points

SEMVER_MINOR = 2
SEMVER_PATCH = 3
SEMVER_LEN = 3


def _parse_version_tag(tag, config):
    tagstring = tag if not isinstance(tag, str) else str(tag)
    match = config.tag_regex.match(tagstring)

    result = None
    if match:
        if len(match.groups()) == 1:
            key = 1
        else:
            key = "version"

        result = {
            "version": match.group(key),
            "prefix": match.group(0)[: match.start(key)],
            "suffix": match.group(0)[match.end(key) :],
        }

    trace(f"tag '{tag}' parsed to {result}")
    return result


def callable_or_entrypoint(group, callable_or_name):
    trace("ep", (group, callable_or_name))

    if callable(callable_or_name):
        return callable_or_name

    for ep in iter_entry_points(group, callable_or_name):
        trace("ep found:", ep.name)
        return ep.load()


def tag_to_version(tag, config=None):
    """
    take a tag that might be prefixed with a keyword and return only the version part
    :param config: optional configuration object
    """
    trace("tag", tag)

    if not config:
        config = Configuration()

    tagdict = _parse_version_tag(tag, config)
    if not isinstance(tagdict, dict) or not tagdict.get("version", None):
        warnings.warn(f"tag {tag!r} no version found")
        return None

    version = tagdict["version"]
    trace("version pre parse", version)

    if tagdict.get("suffix", ""):
        warnings.warn(
            "tag {!r} will be stripped of its suffix '{}'".format(
                tag, tagdict["suffix"]
            )
        )

    version = Version(version)
    trace("version", repr(version))

    return version


def tags_to_versions(tags, config=None):
    """
    take tags that might be prefixed with a keyword and return only the version part
    :param tags: an iterable of tags
    :param config: optional configuration object
    """
    result = []
    for tag in tags:
        tag = tag_to_version(tag, config=config)
        if tag:
            result.append(tag)
    return result


class ScmVersion:
    def __init__(
        self,
        tag_version,
        distance=None,
        node=None,
        dirty=False,
        preformatted=False,
        branch=None,
        config=None,
        node_date=None,
        **kw,
    ):
        if kw:
            trace("unknown args", kw)
        self.tag = tag_version
        if dirty and distance is None:
            distance = 0
        self.distance = distance
        self.node = node
        self.node_date = node_date
        self.time = datetime.datetime.utcfromtimestamp(
            int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
        )
        self._extra = kw
        self.dirty = dirty
        self.preformatted = preformatted
        self.branch = branch
        self.config = config

    @property
    def extra(self):
        warnings.warn(
            "ScmVersion.extra is deprecated and will be removed in future",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._extra

    @property
    def exact(self):
        return self.distance is None

    def __repr__(self):
        return self.format_with(
            "<ScmVersion {tag} d={distance} n={node} d={dirty} b={branch}>"
        )

    def format_with(self, fmt, **kw):
        return fmt.format(
            time=self.time,
            tag=self.tag,
            distance=self.distance,
            node=self.node,
            dirty=self.dirty,
            branch=self.branch,
            node_date=self.node_date,
            **kw,
        )

    def format_choice(self, clean_format, dirty_format, **kw):
        return self.format_with(dirty_format if self.dirty else clean_format, **kw)

    def format_next_version(self, guess_next, fmt="{guessed}.dev{distance}", **kw):
        guessed = guess_next(self.tag, **kw)
        return self.format_with(fmt, guessed=guessed)


def _parse_tag(tag, preformatted, config):
    if preformatted:
        return tag
    if not isinstance(tag, Version):
        tag = tag_to_version(tag, config)
    return tag


def meta(
    tag,
    distance=None,
    dirty=False,
    node=None,
    preformatted=False,
    branch=None,
    config=None,
    **kw,
):
    if not config:
        warnings.warn(
            "meta invoked without explicit configuration,"
            " will use defaults where required."
        )
    parsed_version = _parse_tag(tag, preformatted, config)
    trace("version", tag, "->", parsed_version)
    assert parsed_version is not None, "Can't parse version %s" % tag
    return ScmVersion(
        parsed_version, distance, node, dirty, preformatted, branch, config, **kw
    )


def guess_next_version(tag_version):
    version = _strip_local(str(tag_version))
    return _bump_dev(version) or _bump_regex(version)


def _strip_local(version_string):
    public, sep, local = version_string.partition("+")
    return public


def _bump_dev(version):
    if ".dev" not in version:
        return

    prefix, tail = version.rsplit(".dev", 1)
    if tail != "0":
        raise ValueError(
            "choosing custom numbers for the `.devX` distance "
            "is not supported.\n "
            "The {version} can't be bumped\n"
            "Please drop the tag or create a new supported one".format(version=version)
        )
    return prefix


def _bump_regex(version):
    match = re.match(r"(.*?)(\d+)$", version)
    if match is None:
        raise ValueError(
            "{version} does not end with a number to bump, "
            "please correct or use a custom version scheme".format(version=version)
        )
    else:
        prefix, tail = match.groups()
        return "%s%d" % (prefix, int(tail) + 1)


def guess_next_dev_version(version):
    if version.exact:
        return version.format_with("{tag}")
    else:
        return version.format_next_version(guess_next_version)


def guess_next_simple_semver(version, retain, increment=True):
    try:
        parts = [int(i) for i in str(version).split(".")[:retain]]
    except ValueError:
        raise ValueError(f"{version} can't be parsed as numeric version")
    while len(parts) < retain:
        parts.append(0)
    if increment:
        parts[-1] += 1
    while len(parts) < SEMVER_LEN:
        parts.append(0)
    return ".".join(str(i) for i in parts)


def simplified_semver_version(version):
    if version.exact:
        return guess_next_simple_semver(version.tag, retain=SEMVER_LEN, increment=False)
    else:
        if version.branch is not None and "feature" in version.branch:
            return version.format_next_version(
                guess_next_simple_semver, retain=SEMVER_MINOR
            )
        else:
            return version.format_next_version(
                guess_next_simple_semver, retain=SEMVER_PATCH
            )


def release_branch_semver_version(version):
    if version.exact:
        return version.format_with("{tag}")
    if version.branch is not None:
        # Does the branch name (stripped of namespace) parse as a version?
        branch_ver = _parse_version_tag(version.branch.split("/")[-1], version.config)
        if branch_ver is not None:
            # Does the branch version up to the minor part match the tag? If not it
            # might be like, an issue number or something and not a version number, so
            # we only want to use it if it matches.
            tag_ver_up_to_minor = str(version.tag).split(".")[:SEMVER_MINOR]
            branch_ver_up_to_minor = branch_ver["version"].split(".")[:SEMVER_MINOR]
            if branch_ver_up_to_minor == tag_ver_up_to_minor:
                # We're in a release/maintenance branch, next is a patch/rc/beta bump:
                return version.format_next_version(guess_next_version)
    # We're in a development branch, next is a minor bump:
    return version.format_next_version(guess_next_simple_semver, retain=SEMVER_MINOR)


def release_branch_semver(version):
    warnings.warn(
        "release_branch_semver is deprecated and will be removed in future. "
        + "Use release_branch_semver_version instead",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return release_branch_semver_version(version)


def no_guess_dev_version(version):
    if version.exact:
        return version.format_with("{tag}")
    else:
        return version.format_with("{tag}.post1.dev{distance}")


def date_ver_match(ver):
    match = re.match(
        (
            r"^(?P<date>(?P<year>\d{2}|\d{4})(?:\.\d{1,2}){2})"
            r"(?:\.(?P<patch>\d*)){0,1}?$"
        ),
        str(ver),
    )
    return match


def guess_next_date_ver(version, node_date=None, date_fmt=None):
    """
    same-day -> patch +1
    other-day -> today

    distance is always added as .devX
    """
    match = date_ver_match(version)
    if match is None:
        raise ValueError(
            "{version} does not correspond to a valid versioning date, "
            "please correct or use a custom version scheme".format(version=version)
        )
    # deduct date format if not provided
    if date_fmt is None:
        date_fmt = "%Y.%m.%d" if len(match.group("year")) == 4 else "%y.%m.%d"
    head_date = node_date or datetime.date.today()
    # compute patch
    tag_date = datetime.datetime.strptime(match.group("date"), date_fmt).date()
    if tag_date == head_date:
        patch = match.group("patch") or "0"
        patch = int(patch) + 1
    else:
        if tag_date > head_date:
            # warn on future times
            warnings.warn(
                "your previous tag  ({}) is ahead your node date ({})".format(
                    tag_date, head_date
                )
            )
        patch = 0
    next_version = "{node_date:{date_fmt}}.{patch}".format(
        node_date=head_date, date_fmt=date_fmt, patch=patch
    )
    # rely on the Version object to ensure consistency (e.g. remove leading 0s)
    # TODO: support for intentionally non-normalized date versions
    next_version = str(Version(next_version))
    return next_version


def calver_by_date(version):
    if version.exact and not version.dirty:
        return version.format_with("{tag}")
    # TODO: move the release-X check to a new scheme
    if version.branch is not None and version.branch.startswith("release-"):
        branch_ver = _parse_version_tag(version.branch.split("-")[-1], version.config)
        if branch_ver is not None:
            ver = branch_ver["version"]
            match = date_ver_match(ver)
            if match:
                return ver
    return version.format_next_version(guess_next_date_ver, node_date=version.node_date)


def _format_local_with_time(version, time_format):

    if version.exact or version.node is None:
        return version.format_choice(
            "", "+d{time:{time_format}}", time_format=time_format
        )
    else:
        return version.format_choice(
            "+{node}", "+{node}.d{time:{time_format}}", time_format=time_format
        )


def get_local_node_and_date(version):
    return _format_local_with_time(version, time_format="%Y%m%d")


def get_local_node_and_timestamp(version, fmt="%Y%m%d%H%M%S"):
    return _format_local_with_time(version, time_format=fmt)


def get_local_dirty_tag(version):
    return version.format_choice("", "+dirty")


def get_no_local_node(_):
    return ""


def postrelease_version(version):
    if version.exact:
        return version.format_with("{tag}")
    else:
        return version.format_with("{tag}.post{distance}")


def _get_ep(group, name):
    for ep in iter_entry_points(group, name):
        trace("ep found:", ep.name)
        return ep.load()


def _iter_version_schemes(entrypoint, scheme_value, _memo=None):
    if _memo is None:
        _memo = set()
    if isinstance(scheme_value, str):
        scheme_value = _get_ep(entrypoint, scheme_value)

    if isinstance(scheme_value, (list, tuple)):
        for variant in scheme_value:
            if variant not in _memo:
                _memo.add(variant)
                yield from _iter_version_schemes(entrypoint, variant, _memo=_memo)
    elif callable(scheme_value):
        yield scheme_value


def _call_version_scheme(version, entypoint, given_value, default):
    for scheme in _iter_version_schemes(entypoint, given_value):
        result = scheme(version)
        if result is not None:
            return result
    return default


def format_version(version, **config):
    trace("scm version", version)
    trace("config", config)
    if version.preformatted:
        return version.tag
    main_version = _call_version_scheme(
        version, "setuptools_scm.version_scheme", config["version_scheme"], None
    )
    trace("version", main_version)
    assert main_version is not None
    local_version = _call_version_scheme(
        version, "setuptools_scm.local_scheme", config["local_scheme"], "+unknown"
    )
    trace("local_version", local_version)
    return main_version + local_version
