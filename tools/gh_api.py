"""Functions for Github API requests."""

import getpass
import json
import os
import re
import sys

import requests

try:
    import requests_cache
except ImportError:
    print("no cache", file=sys.stderr)
else:
    requests_cache.install_cache("gh_api", expire_after=3600)

# Keyring stores passwords by a 'username', but we're not storing a username and
# password
fake_username = 'ipython_tools'

class Obj(dict):
    """Dictionary with attribute access to names."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, val):
        self[name] = val

token = None
def get_auth_token():
    global token

    if token is not None:
        return token

    try:
        with open(os.path.join(os.path.expanduser('~'), '.ghoauth')) as f:
            token, = f
            return token
    except Exception:
        pass

    import keyring
    token = keyring.get_password('github', fake_username)
    if token is not None:
        return token

    print("Please enter your github username and password. These are not "
           "stored, only used to get an oAuth token. You can revoke this at "
           "any time on Github.")
    user = input("Username: ")
    pw = getpass.getpass("Password: ")

    auth_request = {
      "scopes": [
        "public_repo",
        "gist"
      ],
      "note": "IPython tools",
      "note_url": "https://github.com/ipython/ipython/tree/master/tools",
    }
    response = requests.post('https://api.github.com/authorizations',
                            auth=(user, pw), data=json.dumps(auth_request))
    response.raise_for_status()
    token = json.loads(response.text)['token']
    keyring.set_password('github', fake_username, token)
    return token

def make_auth_header():
    return {'Authorization': 'token ' + get_auth_token().replace("\n","")}

def post_issue_comment(project, num, body):
    url = 'https://api.github.com/repos/{project}/issues/{num}/comments'.format(project=project, num=num)
    payload = json.dumps({'body': body})
    requests.post(url, data=payload, headers=make_auth_header())

def post_gist(content, description='', filename='file', auth=False):
    """Post some text to a Gist, and return the URL."""
    post_data = json.dumps({
      "description": description,
      "public": True,
      "files": {
        filename: {
          "content": content
        }
      }
    }).encode('utf-8')

    headers = make_auth_header() if auth else {}
    response = requests.post("https://api.github.com/gists", data=post_data, headers=headers)
    response.raise_for_status()
    response_data = json.loads(response.text)
    return response_data['html_url']

def get_pull_request(project, num, auth=False):
    """get pull request info  by number
    """
    url = "https://api.github.com/repos/{project}/pulls/{num}".format(project=project, num=num)
    if auth:
        header = make_auth_header()
    else:
        header = None
    print("fetching %s" % url, file=sys.stderr)
    response = requests.get(url, headers=header)
    response.raise_for_status()
    return json.loads(response.text, object_hook=Obj)

def get_pull_request_files(project, num, auth=False):
    """get list of files in a pull request"""
    url = "https://api.github.com/repos/{project}/pulls/{num}/files".format(project=project, num=num)
    if auth:
        header = make_auth_header()
    else:
        header = None
    return get_paged_request(url, headers=header)

element_pat = re.compile(r'<(.+?)>')
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')

def get_paged_request(url, headers=None, **params):
    """get a full list, handling APIv3's paging"""
    results = []
    params.setdefault("per_page", 100)
    while True:
        if '?' in url:
            params = None
            print("fetching %s" % url, file=sys.stderr)
        else:
            print("fetching %s with %s" % (url, params), file=sys.stderr)
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results.extend(response.json())
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    return results

def get_pulls_list(project, auth=False, **params):
    """get pull request list"""
    params.setdefault("state", "closed")
    url = "https://api.github.com/repos/{project}/pulls".format(project=project)
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    pages = get_paged_request(url, headers=headers, **params)
    return pages

def get_issues_list(project, auth=False, **params):
    """get issues list"""
    params.setdefault("state", "closed")
    url = "https://api.github.com/repos/{project}/issues".format(project=project)
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    pages = get_paged_request(url, headers=headers, **params)
    return pages

def get_milestones(project, auth=False, **params):
    params.setdefault('state', 'all')
    url = "https://api.github.com/repos/{project}/milestones".format(project=project)
    if auth:
        headers = make_auth_header()
    else:
        headers = None
    milestones = get_paged_request(url, headers=headers, **params)
    return milestones

def get_milestone_id(project, milestone, auth=False, **params):
    milestones = get_milestones(project, auth=auth, **params)
    for mstone in milestones:
        if mstone['title'] == milestone:
            return mstone['number']
    raise ValueError("milestone %s not found" % milestone)

def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return bool(issue.get('pull_request', {}).get('html_url', None))

def get_authors(pr):
    print("getting authors for #%i" % pr['number'], file=sys.stderr)
    h = make_auth_header()
    r = requests.get(pr['commits_url'], headers=h)
    r.raise_for_status()
    commits = r.json()
    authors = []
    for commit in commits:
        author = commit['commit']['author']
        authors.append("%s <%s>" % (author['name'], author['email']))
    return authors

# encode_multipart_formdata is from urllib3.filepost
# The only change is to iter_fields, to enforce S3's required key ordering

def iter_fields(fields):
    fields = fields.copy()
    for key in [
            'key', 'acl', 'Filename', 'success_action_status',
            'AWSAccessKeyId', 'Policy', 'Signature', 'Content-Type', 'file']:
        yield key, fields.pop(key)
    yield from fields.items()

def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data mime format.

    :param fields:
        Dictionary of fields or list of (key, value) field tuples.  The key is
        treated as the field name, and the value as the body of the form-data
        bytes. If the value is a tuple of two elements, then the first element
        is treated as the filename of the form-data section.

        Field names and filenames must be unicode.

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`mimetools.choose_boundary`.
    """
    # copy requests imports in here:
    from io import BytesIO
    from requests.packages.urllib3.filepost import (
        choose_boundary, writer, b, get_content_type
    )
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()

    for fieldname, value in iter_fields(fields):
        body.write(b('--%s\r\n' % (boundary)))

        if isinstance(value, tuple):
            filename, data = value
            writer(body).write('Content-Disposition: form-data; name="%s"; '
                               'filename="%s"\r\n' % (fieldname, filename))
            body.write(b('Content-Type: %s\r\n\r\n' %
                       (get_content_type(filename))))
        else:
            data = value
            writer(body).write('Content-Disposition: form-data; name="%s"\r\n'
                               % (fieldname))
            body.write(b'Content-Type: text/plain\r\n\r\n')

        if isinstance(data, int):
            data = str(data)  # Backwards compatibility
        if isinstance(data, str):
            writer(body).write(data)
        else:
            body.write(data)

        body.write(b'\r\n')

    body.write(b('--%s--\r\n' % (boundary)))

    content_type = b('multipart/form-data; boundary=%s' % boundary)

    return body.getvalue(), content_type


def post_download(project, filename, name=None, description=""):
    """Upload a file to the GitHub downloads area"""
    if name is None:
        name = os.path.basename(filename)
    with open(filename, 'rb') as f:
        filedata = f.read()

    url = "https://api.github.com/repos/{project}/downloads".format(project=project)

    payload = json.dumps(dict(name=name, size=len(filedata),
                    description=description))
    response = requests.post(url, data=payload, headers=make_auth_header())
    response.raise_for_status()
    reply = json.loads(response.content)
    s3_url = reply['s3_url']

    fields = dict(
        key=reply['path'],
        acl=reply['acl'],
        success_action_status=201,
        Filename=reply['name'],
        AWSAccessKeyId=reply['accesskeyid'],
        Policy=reply['policy'],
        Signature=reply['signature'],
        file=(reply['name'], filedata),
    )
    fields['Content-Type'] = reply['mime_type']
    data, content_type = encode_multipart_formdata(fields)
    s3r = requests.post(s3_url, data=data, headers={'Content-Type': content_type})
    return s3r
