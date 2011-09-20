"""Demonstrate the Sankey class.

Accepts an optional argument indicating the example to produce (1 through 5). If
the argument isn't provided (or is 0), all the examples will be generated.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib.sankey import Sankey
from itertools import cycle


# Read the optional argument indentifying the example to run.
try:
    example_num = int(sys.argv[1])
except:
    example_num = 0

# Example 1 -- Mostly defaults
# This demonstrates how to create a simple diagram by implicitly calling the
# Sankey.add() method and by appending finish() to the call to the class.
if example_num == 1 or example_num == 0:
    Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
           labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
           orientations=[-1, 1, 0, 1, 1, 1, 0, -1]).finish()
    plt.title("The default settings produce a diagram like this.")
    # Notice:
    #   1. Axes weren't provided when Sankey() was instantiated, so they were
    #      created automatically.
    #   2. The scale argument wasn't necessary since the data was already
    #      normalized.
    #   3. By default, the lengths of the paths are justified.

# Example 2
# This demonstrates:
#   1. Setting one path longer than the others
#   2. Placing a label in the middle of the diagram
#   3. Using the the scale argument to normalize the flows
#   4. Implicitly passing keyword arguments to PathPatch()
#   5. Changing the angle of the arrow heads
#   6. Changing the offset between the tips of the paths and their labels
#   7. Formatting the numbers in the path labels and the associated unit
#   8. Changing the appearance of the patch and the labels after the figure is
#      created
if example_num == 2 or example_num == 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                         title="Flow Diagram of a Widget")
    sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
                    format='%.0f', unit='%')
    sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
               labels = ['', '', '', 'First', 'Second', 'Third', 'Fourth',
                         'Fifth', 'Hurray!'],
               orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
               pathlengths = [0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
                              0.25],
               patchlabel="Widget\nA",
               alpha=0.2, lw=2.0) # Arguments to matplotlib.patches.PathPatch()
    diagrams = sankey.finish()
    diagrams[0].patch.set_facecolor('#37c959')
    diagrams[0].texts[-1].set_color('r')
    diagrams[0].text.set_fontweight('bold')
    # Notice:
    #   1. Since the sum of the flows is nonzero, the width of the trunk isn't
    #      uniform.  If verbose.level is helpful (in matplotlibrc), a message is
    #      given in the terminal window.
    #   2. The second flow doesn't appear because its value is zero.  Again, if
    #      verbose.level is helpful, a message is given in the terminal window.

# Example 3
# This demonstrates:
#   1. Connecting two systems
#   2. Turning off the labels of the quantities
#   3. Adding a legend
if example_num == 3 or example_num == 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
    flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
    sankey = Sankey(ax=ax, unit=None)
    sankey.add(flows=flows, label='one',
               orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
    sankey.add(flows=[-0.25, 0.15, 0.1], fc='#37c959', label='two',
               orientations=[-1, -1, -1], prior=0, connect=(0, 0))
    diagrams = sankey.finish()
    diagrams[-1].patch.set_hatch('/')
    plt.legend(loc='best')
    # Notice that only one connection is specified, but the systems form a
    # circuit since: (1) the lengths of the paths are justified and (2) the
    # orientation and ordering of the flows is mirrored.

# Example 4
# This tests a long chain of connections.
if example_num == 4 or example_num == 0:
    links_per_side = 6
    def side(sankey, n=1):
        """Generate a side chain.
        """
        prior = len(sankey.diagrams)
        colors = cycle(['orange', 'b', 'g', 'r', 'c', 'm', 'y'])
        for i in range(0, 2*n, 2):
            sankey.add(flows=[1, -1], orientations=[-1, -1],
                       patchlabel=str(prior+i), facecolor=colors.next(),
                       prior=prior+i-1, connect=(1, 0), alpha=0.5)
            sankey.add(flows=[1, -1], orientations=[1, 1],
                       patchlabel=str(prior+i+1), facecolor=colors.next(),
                       prior=prior+i, connect=(1, 0), alpha=0.5)
    def corner(sankey):
        """Generate a corner link.
        """
        prior = len(sankey.diagrams)
        sankey.add(flows=[1, -1], orientations=[0, 1],
                   patchlabel=str(prior), facecolor='k',
                   prior=prior-1, connect=(1, 0), alpha=0.5)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                         title="Why would you want to do this?\n(But you could.)")
    sankey = Sankey(ax=ax, unit=None)
    sankey.add(flows=[1, -1], orientations=[0, 1],
               patchlabel="0", facecolor='k',
               rotation=45)
    side(sankey, n=links_per_side)
    corner(sankey)
    side(sankey, n=links_per_side)
    corner(sankey)
    side(sankey, n=links_per_side)
    corner(sankey)
    side(sankey, n=links_per_side)
    sankey.finish()
    # Notice:
    # 1. The alignment doesn't drift significantly (if at all; with 16007
    #    subdiagrams there is still closure).
    # 2. The first diagram is rotated 45 deg, so all other diagrams are rotated
    #    accordingly.

# Example 5
# This is a practical example of a Rankine power cycle.
if example_num == 5 or example_num == 0:
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                         title="Rankine Power Cycle: Example 8.6 from Moran and Shapiro\n"
                               + "\x22Fundamentals of Engineering Thermodynamics\x22, 6th ed., 2008")
    Hdot = [260.431, 35.078, 180.794, 221.115, 22.700,
            142.361, 10.193, 10.210, 43.670, 44.312,
            68.631, 10.758, 10.758, 0.017, 0.642,
            232.121, 44.559, 100.613, 132.168] # MW
    sankey = Sankey(ax=ax, format='%.3G', unit=' MW', gap=0.5, scale=1.0/Hdot[0])
    sankey.add(patchlabel='\n\nPump 1', rotation=90, facecolor='#37c959',
               flows=[Hdot[13], Hdot[6], -Hdot[7]],
               labels=['Shaft power', '', None],
               pathlengths=[0.4, 0.883, 0.25],
               orientations=[1, -1, 0])
    sankey.add(patchlabel='\n\nOpen\nheater', facecolor='#37c959',
               flows=[Hdot[11], Hdot[7], Hdot[4], -Hdot[8]],
               labels=[None, '', None, None],
               pathlengths=[0.25, 0.25, 1.93, 0.25],
               orientations=[1, 0, -1, 0], prior=0, connect=(2, 1))
    sankey.add(patchlabel='\n\nPump 2', facecolor='#37c959',
               flows=[Hdot[14], Hdot[8], -Hdot[9]],
               labels=['Shaft power', '', None],
               pathlengths=[0.4, 0.25, 0.25],
               orientations=[1, 0, 0], prior=1, connect=(3, 1))
    sankey.add(patchlabel='Closed\nheater', trunklength=2.914, fc='#37c959',
               flows=[Hdot[9], Hdot[1], -Hdot[11], -Hdot[10]],
               pathlengths=[0.25, 1.543, 0.25, 0.25],
               labels=['', '', None, None],
               orientations=[0, -1, 1, -1], prior=2, connect=(2, 0))
    sankey.add(patchlabel='Trap', facecolor='#37c959', trunklength=5.102,
               flows=[Hdot[11], -Hdot[12]],
               labels=['\n', None],
               pathlengths=[1.0, 1.01],
               orientations=[1, 1], prior=3, connect=(2, 0))
    sankey.add(patchlabel='Steam\ngenerator', facecolor='#ff5555',
               flows=[Hdot[15], Hdot[10], Hdot[2], -Hdot[3], -Hdot[0]],
               labels=['Heat rate', '', '', None, None],
               pathlengths=0.25,
               orientations=[1, 0, -1, -1, -1], prior=3, connect=(3, 1))
    sankey.add(patchlabel='\n\n\nTurbine 1', facecolor='#37c959',
               flows=[Hdot[0], -Hdot[16], -Hdot[1], -Hdot[2]],
               labels=['', None, None, None],
               pathlengths=[0.25, 0.153, 1.543, 0.25],
               orientations=[0, 1, -1, -1], prior=5, connect=(4, 0))
    sankey.add(patchlabel='\n\n\nReheat', facecolor='#37c959',
               flows=[Hdot[2], -Hdot[2]],
               labels=[None, None],
               pathlengths=[0.725, 0.25],
               orientations=[-1, 0], prior=6, connect=(3, 0))
    sankey.add(patchlabel='Turbine 2', trunklength=3.212, facecolor='#37c959',
               flows=[Hdot[3], Hdot[16], -Hdot[5], -Hdot[4], -Hdot[17]],
               labels=[None, 'Shaft power', None, '', 'Shaft power'],
               pathlengths=[0.751, 0.15, 0.25, 1.93, 0.25],
               orientations=[0, -1, 0, -1, 1], prior=6, connect=(1, 1))
    sankey.add(patchlabel='Condenser', facecolor='#58b1fa', trunklength=1.764,
               flows=[Hdot[5], -Hdot[18], -Hdot[6]],
               labels=['', 'Heat rate', None],
               pathlengths=[0.45, 0.25, 0.883],
               orientations=[-1, 1, 0], prior=8, connect=(2, 0))
    diagrams = sankey.finish()
    for diagram in diagrams:
        diagram.text.set_fontweight('bold')
        diagram.text.set_fontsize('10')
        for text in diagram.texts:
            text.set_fontsize('10')
    # Notice that the explicit connections are handled automatically, but the
    # implicit ones currently are not.  The lengths of the paths and the trunks
    # must be adjusted manually, and that is a bit tricky.

plt.show()
