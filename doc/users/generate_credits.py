#!/usr/bin/env python
#
# This script generates credits.rst with an up-to-date list of contributors
# to the matplotlib github repository.

import subprocess

TEMPLATE = """.. Note: This file is auto-generated using generate_credits.py

.. _credits:

*******
Credits
*******


Matplotlib was written by John D. Hunter, with contributions from
an ever-increasing number of users and developers.
The current co-lead developers are Michael Droettboom
and Thomas A. Caswell; they are assisted by many
`active
<https://www.openhub.net/p/matplotlib/contributors>`_ developers.

The following is a list of contributors extracted from the
git revision control history of the project:

{contributors}

Some earlier contributors not included above are (with apologies
to any we have missed):

Charles Twardy,
Gary Ruben,
John Gill,
David Moore,
Paul Barrett,
Jared Wahlstrand,
Jim Benson,
Paul Mcguire,
Andrew Dalke,
Nadia Dencheva,
Baptiste Carvello,
Sigve Tjoraand,
Ted Drain,
James Amundson,
Daishi Harada,
Nicolas Young,
Paul Kienzle,
John Porter,
and Jonathon Taylor.

We also thank all who have reported bugs, commented on
proposed changes, or otherwise contributed to Matplotlib's
development and usefulness.
"""


def main():
    text = subprocess.check_output(['git', 'shortlog', '--summary'])
    contributors = [line.split('\t', 1)[1].strip()
                    for line in text.decode('utf8').split('\n')
                    if line]
    with open('credits.rst', 'w') as f:
        f.write(TEMPLATE.format(contributors=',\n'.join(contributors)))


if __name__ == '__main__':
    main()