#!/usr/bin/env python
"""
Gets the data from failed image comparison tests from a Travis job.

Usage:

    ./get_travis_results.py job_id

The job id is displayed in the URL when viewing a job on the Travis
web interface.

This will download the data and extract it into a directory
"result_images" in the current directory.
"""


import base64
import io
import json
import re
import tarfile
import urllib2


def get_travis_results(id):
    url = "http://travis-ci.org/jobs/{0}.json".format(id)
    print "Retrieving URL:", url
    f = urllib2.urlopen(url)
    payload = f.read()
    f.close()
    print "Done"

    job = json.loads(payload)

    log = job['log']
    match = re.search(">>>>>>>>TARBALL>>>>>>>>", log)
    if match is None:
        print("No test data payload found")
        return

    log = log[match.end():].strip()

    log = base64.b64decode(log)

    data = io.BytesIO(log)
    data.seek(0)
    with tarfile.open(mode="r|bz2", fileobj=data) as tar:
        tar.extractall()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Get the test results from a Travis job')
    parser.add_argument('id', type=int, nargs=1, help='The job id')

    args = parser.parse_args()
    get_travis_results(args.id[0])
