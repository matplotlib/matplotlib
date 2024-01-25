#!/usr/bin/env python
"""
Simple tools to query github.com and gather stats about issues.

To generate a report for Matplotlib 3.0.0, run:

    python github_stats.py --milestone 3.0.0 --since-tag v2.0.0
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys

from argparse import ArgumentParser
from datetime import datetime, timedelta
from subprocess import check_output

from gh_api import (
    get_paged_request, make_auth_header, get_pull_request, is_pull_request,
    get_milestone_id, get_issues_list, get_authors,
)
# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

REPORT_TEMPLATE = """\
.. _github-stats:

{title}
{title_underline}

GitHub statistics for {since_day} (tag: {tag}) - {today}

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed {n_issues} issues and merged {n_pulls} pull requests.
{milestone}
The following {nauthors} authors contributed {ncommits} commits.

{unique_authors}
{links}

Previous GitHub statistics
--------------------------

.. toctree::
    :maxdepth: 1
    :glob:
    :reversed:

    prev_whats_new/github_stats_*"""
MILESTONE_TEMPLATE = (
    'The full list can be seen `on GitHub '
    '<https://github.com/{project}/milestone/{milestone_id}?closed=1>`__\n')
LINKS_TEMPLATE = """
GitHub issues and pull requests:

Pull Requests ({n_pulls}):

{pull_request_report}

Issues ({n_issues}):

{issue_report}
"""

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def round_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def _parse_datetime(s):
    """Parse dates in the format returned by the GitHub API."""
    return datetime.strptime(s, ISO8601) if s else datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    return {i['number']: i for i in issues}


def split_pulls(all_issues, project="matplotlib/matplotlib"):
    """Split a list of closed issues into non-PR Issues and Pull Requests."""
    pulls = []
    issues = []
    for i in all_issues:
        if is_pull_request(i):
            pull = get_pull_request(project, i['number'], auth=True)
            pulls.append(pull)
        else:
            issues.append(i)
    return issues, pulls


def issues_closed_since(period=timedelta(days=365),
                        project='matplotlib/matplotlib', pulls=False):
    """
    Get all issues closed since a particular point in time.

    *period* can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present.
    """

    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        since = round_hour(datetime.utcnow() - period)
    else:
        since = period
    url = (
        f'https://api.github.com/repos/{project}/{which}'
        f'?state=closed'
        f'&sort=updated'
        f'&since={since.strftime(ISO8601)}'
        f'&per_page={PER_PAGE}')
    allclosed = get_paged_request(url, headers=make_auth_header())

    filtered = (i for i in allclosed
                if _parse_datetime(i['closed_at']) > since)
    if pulls:
        filtered = (i for i in filtered
                    if _parse_datetime(i['merged_at']) > since)
        # filter out PRs not against main (backports)
        filtered = (i for i in filtered if i['base']['ref'] == 'main')
    else:
        filtered = (i for i in filtered if not is_pull_request(i))

    return list(filtered)


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date."""
    return sorted(issues, key=lambda i: i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title."""
    lines = []
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            number = i['number']
            title = i['title'].replace('`', '``').strip()
            lines.append(f'* :{role}:`{number}`: {title}')
    else:
        for i in issues:
            number = i['number']
            title = i['title'].replace('`', '``').strip()
            lines.append('* {number}: {title}')
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Whether to add reST urls for all issues in printout.
    show_urls = True

    parser = ArgumentParser()
    parser.add_argument(
        '--since-tag', type=str,
        help='The git tag to use for the starting point '
             '(typically the last macro release).')
    parser.add_argument(
        '--milestone', type=str,
        help='The GitHub milestone to use for filtering issues [optional].')
    parser.add_argument(
        '--days', type=int,
        help='The number of days of data to summarize (use this or --since-tag).')
    parser.add_argument(
        '--project', type=str, default='matplotlib/matplotlib',
        help='The project to summarize.')
    parser.add_argument(
        '--links', action='store_true', default=False,
        help='Include links to all closed Issues and PRs in the output.')

    opts = parser.parse_args()
    tag = opts.since_tag

    # set `since` from days or git tag
    if opts.days:
        since = datetime.utcnow() - timedelta(days=opts.days)
    else:
        if not tag:
            tag = check_output(['git', 'describe', '--abbrev=0'],
                               encoding='utf8').strip()
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        tagday, tz = check_output(cmd, encoding='utf8').strip().rsplit(' ', 1)
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
        h = int(tz[1:3])
        m = int(tz[3:])
        td = timedelta(hours=h, minutes=m)
        if tz[0] == '-':
            since += td
        else:
            since -= td

    since = round_hour(since)

    milestone = opts.milestone
    project = opts.project

    print(f'fetching GitHub stats since {since} (tag: {tag}, milestone: {milestone})',
          file=sys.stderr)
    if milestone:
        milestone_id = get_milestone_id(project=project, milestone=milestone,
                                        auth=True)
        issues_and_pulls = get_issues_list(project=project, milestone=milestone_id,
                                           state='closed', auth=True)
        issues, pulls = split_pulls(issues_and_pulls, project=project)
    else:
        issues = issues_closed_since(since, project=project, pulls=False)
        pulls = issues_closed_since(since, project=project, pulls=True)

    # For regular reports, it's nice to show them in reverse chronological order.
    issues = sorted_by_field(issues, reverse=True)
    pulls = sorted_by_field(pulls, reverse=True)

    n_issues, n_pulls = map(len, (issues, pulls))
    n_total = n_issues + n_pulls
    since_day = since.strftime("%Y/%m/%d")
    today = datetime.today()

    title = (f'GitHub statistics for {milestone.lstrip("v")} '
             f'{today.strftime("(%b %d, %Y)")}')

    ncommits = 0
    all_authors = []
    if tag:
        # print git info, in addition to GitHub info:
        since_tag = f'{tag}..'
        cmd = ['git', 'log', '--oneline', since_tag]
        ncommits += len(check_output(cmd).splitlines())

        author_cmd = ['git', 'log', '--use-mailmap', '--format=* %aN', since_tag]
        all_authors.extend(
            check_output(author_cmd, encoding='utf-8', errors='replace').splitlines())

    pr_authors = []
    for pr in pulls:
        pr_authors.extend(get_authors(pr))
    ncommits = len(pr_authors) + ncommits - len(pulls)
    author_cmd = ['git', 'check-mailmap'] + pr_authors
    with_email = check_output(author_cmd,
                              encoding='utf-8', errors='replace').splitlines()
    all_authors.extend(['* ' + a.split(' <')[0] for a in with_email])
    unique_authors = sorted(set(all_authors), key=lambda s: s.lower())

    if milestone:
        milestone_str = MILESTONE_TEMPLATE.format(project=project,
                                                  milestone_id=milestone_id)
    else:
        milestone_str = ''

    if opts.links:
        links = LINKS_TEMPLATE.format(n_pulls=n_pulls,
                                      pull_request_report=report(pulls, show_urls),
                                      n_issues=n_issues,
                                      issue_report=report(issues, show_urls))
    else:
        links = ''

    # Print summary report we can directly include into release notes.
    print(REPORT_TEMPLATE.format(title=title, title_underline='=' * len(title),
                                 since_day=since_day, tag=tag,
                                 today=today.strftime('%Y/%m/%d'),
                                 n_issues=n_issues, n_pulls=n_pulls,
                                 milestone=milestone_str,
                                 nauthors=len(unique_authors), ncommits=ncommits,
                                 unique_authors='\n'.join(unique_authors), links=links))
