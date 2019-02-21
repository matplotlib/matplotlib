#!/usr/bin/env python
"""Simple tools to query github.com and gather stats about issues.

To generate a report for IPython 2.0, run:

    python github_stats.py --milestone 2.0 --since-tag rel-1.0.0
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import codecs
import sys

from argparse import ArgumentParser
from datetime import datetime, timedelta
from subprocess import check_output

from gh_api import (
    get_paged_request, make_auth_header, get_pull_request, is_pull_request,
    get_milestone_id, get_issues_list, get_authors,
)
#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def round_hour(dt):
    return dt.replace(minute=0,second=0,microsecond=0)

def _parse_datetime(s):
    """Parse dates in the format returned by the Github API."""
    if s:
        return datetime.strptime(s, ISO8601)
    else:
        return datetime.fromtimestamp(0)

def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    idict = {}
    for i in issues:
        idict[i['number']] = i
    return idict

def split_pulls(all_issues, project="ipython/ipython"):
    """split a list of closed issues into non-PR Issues and Pull Requests"""
    pulls = []
    issues = []
    for i in all_issues:
        if is_pull_request(i):
            pull = get_pull_request(project, i['number'], auth=True)
            pulls.append(pull)
        else:
            issues.append(i)
    return issues, pulls


def issues_closed_since(period=timedelta(days=365), project="ipython/ipython", pulls=False):
    """Get all issues closed since a particular point in time. period
    can either be a datetime object, or a timedelta object. In the
    latter case, it is used as a time before the present.
    """

    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        since = round_hour(datetime.utcnow() - period)
    else:
        since = period
    url = "https://api.github.com/repos/%s/%s?state=closed&sort=updated&since=%s&per_page=%i" % (project, which, since.strftime(ISO8601), PER_PAGE)
    allclosed = get_paged_request(url, headers=make_auth_header())

    filtered = [ i for i in allclosed if _parse_datetime(i['closed_at']) > since ]
    if pulls:
        filtered = [ i for i in filtered if _parse_datetime(i['merged_at']) > since ]
        # filter out PRs not against master (backports)
        filtered = [ i for i in filtered if i['base']['ref'] == 'master' ]
    else:
        filtered = [ i for i in filtered if not is_pull_request(i) ]

    return filtered


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key = lambda i:i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title."""
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged_at' in i else 'ghissue'
            print('* :%s:`%d`: %s' % (role, i['number'],
                                      i['title'].replace('`', '``')))
    else:
        for i in issues:
            print('* %d: %s' % (i['number'], i['title'].replace('`', '``')))

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    # Whether to add reST urls for all issues in printout.
    show_urls = True

    parser = ArgumentParser()
    parser.add_argument('--since-tag', type=str,
        help="The git tag to use for the starting point (typically the last major release)."
    )
    parser.add_argument('--milestone', type=str,
        help="The GitHub milestone to use for filtering issues [optional]."
    )
    parser.add_argument('--days', type=int,
        help="The number of days of data to summarize (use this or --since-tag)."
    )
    parser.add_argument('--project', type=str, default="ipython/ipython",
        help="The project to summarize."
    )
    parser.add_argument('--links', action='store_true', default=False,
        help="Include links to all closed Issues and PRs in the output."
    )

    opts = parser.parse_args()
    tag = opts.since_tag

    # set `since` from days or git tag
    if opts.days:
        since = datetime.utcnow() - timedelta(days=opts.days)
    else:
        if not tag:
            tag = check_output(['git', 'describe', '--abbrev=0']).strip().decode('utf8')
        cmd = ['git', 'log', '-1', '--format=%ai', tag]
        tagday, tz = check_output(cmd).strip().decode('utf8').rsplit(' ', 1)
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

    print("fetching GitHub stats since %s (tag: %s, milestone: %s)" % (since, tag, milestone), file=sys.stderr)
    if milestone:
        milestone_id = get_milestone_id(project=project, milestone=milestone,
                auth=True)
        issues_and_pulls = get_issues_list(project=project,
                milestone=milestone_id,
                state='closed',
                auth=True,
        )
        issues, pulls = split_pulls(issues_and_pulls, project=project)
    else:
        issues = issues_closed_since(since, project=project, pulls=False)
        pulls = issues_closed_since(since, project=project, pulls=True)

    # For regular reports, it's nice to show them in reverse chronological order
    issues = sorted_by_field(issues, reverse=True)
    pulls = sorted_by_field(pulls, reverse=True)

    n_issues, n_pulls = map(len, (issues, pulls))
    n_total = n_issues + n_pulls

    # Print summary report we can directly include into release notes.
    print('.. _github-stats:')
    print()
    print('GitHub Stats')
    print('============')

    print()
    since_day = since.strftime("%Y/%m/%d")
    today = datetime.today().strftime("%Y/%m/%d")
    print("GitHub stats for %s - %s (tag: %s)" % (since_day, today, tag))
    print()
    print("These lists are automatically generated, and may be incomplete or contain duplicates.")
    print()

    ncommits = 0
    all_authors = []
    if tag:
        # print git info, in addition to GitHub info:
        since_tag = tag+'..'
        cmd = ['git', 'log', '--oneline', since_tag]
        ncommits += len(check_output(cmd).splitlines())

        author_cmd = ['git', 'log', '--use-mailmap', "--format=* %aN", since_tag]
        all_authors.extend(check_output(author_cmd).decode('utf-8', 'replace').splitlines())

    pr_authors = []
    for pr in pulls:
        pr_authors.extend(get_authors(pr))
    ncommits = len(pr_authors) + ncommits - len(pulls)
    author_cmd = ['git', 'check-mailmap'] + pr_authors
    with_email = check_output(author_cmd).decode('utf-8', 'replace').splitlines()
    all_authors.extend(['* ' + a.split(' <')[0] for a in with_email])
    unique_authors = sorted(set(all_authors), key=lambda s: s.lower())

    print("We closed %d issues and merged %d pull requests." % (n_issues, n_pulls))
    if milestone:
        print("The full list can be seen `on GitHub <https://github.com/%s/milestone/%s>`__"
            % (project, milestone)
        )

    print()
    print("The following %i authors contributed %i commits." % (len(unique_authors), ncommits))
    print()
    print('\n'.join(unique_authors))

    if opts.links:
        print()
        print("GitHub issues and pull requests:")
        print()
        print('Pull Requests (%d):\n' % n_pulls)
        report(pulls, show_urls)
        print()
        print('Issues (%d):\n' % n_issues)
        report(issues, show_urls)
