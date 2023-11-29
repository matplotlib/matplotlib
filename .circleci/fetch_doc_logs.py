"""
Download artifacts from CircleCI for a documentation build.

This is run by the :file:`.github/workflows/circleci.yml` workflow in order to
get the warning/deprecation logs that will be posted on commits as checks. Logs
are downloaded from the :file:`docs/logs` artifact path and placed in the
:file:`logs` directory.

Additionally, the artifact count for a build is produced as a workflow output,
by appending to the file specified by :env:`GITHUB_OUTPUT`.

If there are no logs, an "ERROR" message is printed, but this is not fatal, as
the initial 'status' workflow runs when the build has first started, and there
are naturally no artifacts at that point.

This script should be run by passing the CircleCI build URL as its first
argument. In the GitHub Actions workflow, this URL comes from
``github.event.target_url``.
"""
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import URLError, urlopen


if len(sys.argv) != 2:
    print('USAGE: fetch_doc_results.py CircleCI-build-url')
    sys.exit(1)

target_url = urlparse(sys.argv[1])
*_, organization, repository, build_id = target_url.path.split('/')
print(f'Fetching artifacts from {organization}/{repository} for {build_id}')

artifact_url = (
    f'https://circleci.com/api/v2/project/gh/'
    f'{organization}/{repository}/{build_id}/artifacts'
)
print(artifact_url)
try:
    with urlopen(artifact_url) as response:
        artifacts = json.load(response)
except URLError:
    artifacts = {'items': []}
artifact_count = len(artifacts['items'])
print(f'Found {artifact_count} artifacts')

with open(os.environ['GITHUB_OUTPUT'], 'w+') as fd:
    fd.write(f'count={artifact_count}\n')

logs = Path('logs')
logs.mkdir(exist_ok=True)

found = False
for item in artifacts['items']:
    path = item['path']
    if path.startswith('doc/logs/'):
        path = Path(path).name
        print(f'Downloading {path} from {item["url"]}')
        with urlopen(item['url']) as response:
            (logs / path).write_bytes(response.read())
        found = True

if not found:
    print('ERROR: Did not find any artifact logs!')
