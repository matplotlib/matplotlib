# print png to standard out
# usage: python print_stdout.py > somefile.png
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.matlab import *

plot([1,2,3])

savefig(sys.stdout)
show()
