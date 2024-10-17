"""
=====================
Print image to stdout
=====================

print png to standard out

usage: python print_stdout.py > somefile.png

"""

import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.savefig(sys.stdout.buffer)
