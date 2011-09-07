#!/usr/bin/env python
"""Module for creating Sankey diagrams using matplotlib
"""
__author__ = "Kevin L. Davies"
__credits__ = ["Yannick Copin"]
__license__ = "BSD"
__version__ = "2011/09/07"
# Original version by Yannick Copin (ycopin@ipnl.in2p3.fr) 10/2/2010, available
# at:
#     http://matplotlib.sourceforge.net/examples/api/sankey_demo.html
# Modifications by Kevin Davies (kld@alumni.carnegiemellon.edu) 6/3/2011:
#   --Used arcs for the curves (so that the widths of the paths are uniform)
#   --Converted the function to a class and created methods to join
#     multiple simple Sankey diagrams
#   --Provided handling for cases where the total of the inputs isn't 100
#     Now, the default layout is based on the assumption that the inputs sum to
#     1.  A scaling parameter can be used in other cases.
#   --The call structure was changed to be more explicit about layout, including
#     the length of the trunk, length of the paths, gap between the paths, and
#     the margin around the diagram.
#   --Allowed the lengths of paths to be adjusted individually, with an option
#     to automatically justify them
#   --The call structure was changed to make the specification of path
#     orientation more flexible.  Flows are passed through one array, with
#     inputs being positive and outputs being negative.  An orientation argment
#     specifies the direction of the arrows.  The "main" inputs/outputs are now
#     specified via an orientation of 0, and there may be several of each.
#   --Added assertions to catch common calling errors
#    -Added the physical unit as a string argument to be used in the labels, so
#     that the values of the flows can usually be applied automatically
#   --Added an argument for a minimum magnitude below which flows are not shown
#   --Added a tapered trunk in the case that the flows do not sum to 0
#   --Allowed the diagram to be rotated

import numpy as np
from matplotlib.cbook import iterable, Bunch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib import verbose
#from collections import namedtuple
# Note: If you cannot use namedtuple (it was introduced in Python 2.6), then
# comment out the line above and switch out the commented code wherever
# "Without namedtuple" is written in the code that follows.

# Angles (in deg/90)
RIGHT = 0
UP = 1
# LEFT = 2
DOWN = 3


# Container class for information about a simple Sankey diagram, i.e., one with
# inputs/outputs at a single hierarchial level
#SankeyInfo = namedtuple('SankeyInfo', 'patch flows angles tips text texts')
# Without namedtuple: Comment out the line above.
# See Sankey.finish() for a description of the fields.


class Sankey:
    """Sankey diagram in matplotlib

    "Sankey diagrams are a specific type of flow diagram, in which the width of
    the arrows is shown proportionally to the flow quantity.  They are typically
    used to visualize energy or material or cost transfers between processes."
    --http://en.wikipedia.org/wiki/Sankey_diagram, accessed 6/1/2011
    """
    def _arc(self, quadrant=0, cw=True, radius=1, center=(0,0)):
        """Return the codes and vertices for a rotated, scaled, and translated
        90 degree arc.

        quadrant:  Uses 0-based indexing (0, 1, 2, or 3)
        cw:        If True, clockwise
        center:    (x, y) tuple of the arc's center

        Note:  It would be possible to use matplotlib's transforms to do this,
        but since the rotations is discrete, it's just as easy and maybe more
        efficient to do it here.
        """
        ARC_CODES = [Path.LINETO,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4]
        # Vertices of a cubic Bezier curve approximating a 90 deg arc
        # These can be determined by Path.arc(0,90).
        ARC_VERTICES = np.array([[1.00000000e+00, 0.00000000e+00],
                                 [1.00000000e+00, 2.65114773e-01],
                                 [8.94571235e-01, 5.19642327e-01],
                                 [7.07106781e-01, 7.07106781e-01],
                                 [5.19642327e-01, 8.94571235e-01],
                                 [2.65114773e-01, 1.00000000e+00],
                                 #[6.12303177e-17, 1.00000000e+00]])
                                 [0.00000000e+00, 1.00000000e+00]])
        if quadrant == 0 or quadrant == 2:
            if cw:
                vertices = ARC_VERTICES
            else:
                vertices = ARC_VERTICES[:,::-1] # Swap x and y
        elif quadrant == 1 or quadrant == 3:
            # Negate x
            if cw:
                # Swap x and y
                vertices = np.column_stack((-ARC_VERTICES[:,1], ARC_VERTICES[:,0]))
            else:
                vertices = np.column_stack((-ARC_VERTICES[:,0], ARC_VERTICES[:,1]))
        if quadrant > 1: radius = -radius # Rotate 180 deg
        return zip(ARC_CODES,
                   radius*vertices + np.tile(center, (ARC_VERTICES.shape[0], 1)))

    def _add_input(self, path, angle, flow, length):
        """Add an input to a path and return its tip and label locations.
        """
        if angle is None:
            return [0, 0], [0, 0]
        else:
            (x, y) = path[-1][1]            # Use the last point as a reference.
            dipdepth = (flow / 2) * self.pitch
            if angle == RIGHT:
                x -= length
                dip = [x + dipdepth, y + flow / 2.0]
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x+self.gap, y + flow])])
                label_location = [dip[0] - self.offset, dip[1]]
            else:                           # Vertical
                x -= self.gap
                if angle==UP: sign = 1
                else: sign = -1

                dip = [x - flow / 2, y - sign * (length - dipdepth)]
                if angle==DOWN: q = 2
                else: q = 1

                if self.radius:   # Inner arc not needed if inner radius is zero
                    path.extend(self._arc(quadrant=q,
                                          cw=angle==UP,
                                          radius=self.radius,
                                          center=(x + self.radius,
                                                  y - sign * self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                path.extend([(Path.LINETO, [x, y - sign * length]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x - flow, y - sign * length])])
                path.extend(self._arc(quadrant=q,
                                      cw=angle==DOWN,
                                      radius=flow + self.radius,
                                      center=(x + self.radius,
                                              y - sign * self.radius)))
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                label_location = [dip[0], dip[1] - sign * self.offset]

            return dip, label_location

    def _add_output(self, path, angle, flow, length):
        """Append an output to a path and return its tip and label locations.

        Note: flow is negative for an output.
        """
        if angle is None:
            return [0, 0], [0, 0]
        else:
            (x, y) = path[-1][1]             # Use the last point as a reference.
            tipheight = (self.shoulder - flow / 2) * self.pitch
            if angle == RIGHT:
                x += length
                tip = [x + tipheight, y + flow / 2.0]
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, [x, y + self.shoulder]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x, y - self.shoulder + flow]),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x-self.gap, y + flow])])
                label_location = [tip[0] + self.offset, tip[1]]
            else:                            # Vertical
                x += self.gap
                if angle==UP: sign = 1
                else: sign = -1

                tip = [x - flow / 2.0, y + sign * (length + tipheight)]
                if angle==UP:
                    q = 3
                else:
                    q = 0
                if self.radius:   # Inner arc not needed if inner radius is zero
                    path.extend(self._arc(quadrant=q,
                                          cw=angle==UP,
                                          radius=self.radius,
                                          center=(x - self.radius,
                                                  y + sign*self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                path.extend([(Path.LINETO, [x, y + sign * length]),
                             (Path.LINETO, [x - self.shoulder, y + sign * length]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x + self.shoulder - flow, y + sign * length]),
                             (Path.LINETO, [x - flow, y + sign * length])])
                path.extend(self._arc(quadrant=q,
                                      cw=angle==DOWN,
                                      radius=self.radius - flow,
                                      center=(x - self.radius,
                                              y + sign * self.radius)))
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                label_location = [tip[0], tip[1] + sign * self.offset]
            return tip, label_location

    def _revert(self, path, first_action=Path.LINETO):
        """A path is not simply revertable by path[::-1] since the code
        specifies an action to take from the _previous_ point.
        """
        reverse_path = []
        next_code = first_action
        for code,position in path[::-1]:
            reverse_path.append((next_code, position))
            next_code = code
        return reverse_path
        # This might be more efficient, but it fails because 'tuple' object
        # doesn't support item assignment:
        #path[1] = path[1][-1:0:-1]
        #path[1][0] = first_action
        #path[2] = path[2][::-1]
        #return path

    def add(self, patchlabel='',
            flows=np.array([1.0,-1.0]), orientations=[0,0], labels='',
            trunklength=1.0, pathlengths=0.25, prior=None, connect=(0,0),
            rotation=0, **kwargs):
        """Add a simple Sankey diagram with flows at the same hierarchial level.

        patchlabel:    Label to be placed at the center of the diagram
                       Note: label (not patchlabel) will be passed the patch
                       through **kwargs below and can be used to create an
                       entry in the legend.
        flows:         Array of flow values
                       By convention, inputs are positive and outputs are
                       negative.
        orientations:  List of orientations of the paths.
                       The values should be 1 (from/to the top), 0 (from/to a
                       the left or right), or -1 (from/to the bottom).  If 0,
                       inputs will break in from the left and outputs will break
                       away to the right.
        labels:        List of specifications of the labels for the flows
                       Each value may be None (no labels), '' (just label the
                       quantities), or a labeling string.  If a single value is
                       provided, it will be applied to all flows.  If an entry
                       is a non-empty string, then the quantity for the
                       corresponding flow will be shown below the string.
                       However, if the unit of the main diagram is None, then
                       quantities are never shown, regardless of the value of
                       this argument.
        trunklength:   Length between the bases of the input and output groups
        pathlengths:   List of lengths of the arrows before break-in or after
                       break-away
                       If a single value is given, then it will be applied to
                       the first (inside) paths on the top and bottom, and the
                       length of all other arrows will be justified accordingly.
                       Ths pathlengths are not applied to the hoizontal inputs
                       and outputs.
        prior:         Index of the prior diagram to which this diagram should
                       be connected
        connect:       A (prior, this) tuple indexing the flow of the prior
                       diagram and the flow of this diagram which should be
                       connected
                       If this is the first diagram or prior is None, connect
                       will be ignored.
        rotation:      Angle of rotation of the diagram [deg]
                       rotation is ignored if this diagram is connected to an
                       existing one (using prior and connect).  The
                       interpretation of the orientations argument will be
                       rotated accordingly (e.g., if rotation == 90, an
                       orientations entry of 1 means to/from the left).
        **kwargs:      Propagated to matplotlib.patches.PathPatch (e.g.,
                       fill=False, label="A legend entry")
                       By default, facecolor='#bfd1d4' (light blue) and
                       lineweight=0.5.

        The indexing parameters (prior and connect) are zero-based.

        The flows are placed along the top of the diagram from the inside out in
        order of their index within the flows list or array.  They are placed
        along the sides of the diagram from the top down and along the bottom
        from the outside in.

        If the the sum of the inputs and outputs is not zero, the discrepancy
        will show as a cubic Bezier curve along the top and bottom edges of the
        trunk.
        """
        # Check and preprocess the arguments.
        flows = np.array(flows)
        n = flows.shape[0]  # Number of flows
        if rotation == None:
            rotation = 0
        else:
            rotation /= 90.0 # In the code below, angles are expressed in deg/90.
        assert len(orientations) == n, \
               "orientations and flows must have the same length.\n" \
               + "orientations has length " + str(len(orientations)) \
               + ", but flows has length " + str(n) + "."
        if getattr(labels, '__iter__', False):
        # iterable() isn't used because it would give True if labels is a string.
            assert len(labels) == n, \
                "If labels is a list, then labels and flows must " \
                + "have the same length.\nlabels has length " \
                + str(len(labels)) + ", but flows has length " + str(n) + "."
        else:
            labels = [labels]*n
        assert trunklength >= 0, \
               "trunklength is negative.\n" \
               + "This isn't allowed because it would cause poor layout."
        if np.absolute(np.sum(flows)) > self.tolerance:
            verbose.report("The sum of the flows is nonzero (" \
               + str(np.sum(flows)) + ").\nIs the system not at steady state?", 
              'helpful')
        scaled_flows = self.scale*flows
        gain = sum(max(flow, 0) for flow in scaled_flows)
        loss = sum(min(flow, 0) for flow in scaled_flows)
        if not (0.5 <= gain <= 2.0):
            verbose.report("The scaled sum of the inputs is " + str(gain) \
                + ".\nThis may cause poor layout.\nConsider changing the " \
                "scale so that the scaled sum is approximately 1.0.", 
                'helpful')
        if not (-2.0 <= loss <= -0.5):
            verbose.report("The scaled sum of the outputs is " + str(gain) \
                + ".\nThis may cause poor layout.\nConsider changing the " \
                "scale so that the scaled sum is approximately 1.0.", 
                'helpful')
        if prior is not None:
            assert prior >= 0, \
                   "The index of the prior diagram is negative."
            assert min(connect) >= 0, \
                   "At least one of the connection indices is negative."
            assert prior < len(self.diagrams), \
                   "The index of the prior diagram is " + str(prior) \
                   + " but there are only " + str(len(self.diagrams)) \
                   + " other diagrams.\nThe index is zero-based."
            assert connect[0] < len(self.diagrams[prior].flows), \
                   "The connection index to the source diagram is " \
                   + str(connect[0]) + " but that diagram has only " \
                   + str(len(self.diagrams[prior].flows)) \
                   + " flows.\nThe index is zero-based."
            # Without namedtuple:
            #assert connect[0] < len(self.diagrams[prior][1]), \
            #       "The connection index to the source diagram is " \
            #       + str(connect[0]) + " but that diagram has only " \
            #       + len(self.diagrams[prior][1]) \
            #       + " flows.\nThe index is zero-based."
            assert connect[1] < n, \
                   "The connection index to this diagram is " \
                   + str(connect[1]) + " but this diagram has only " \
                   + str(n) + " flows.\nThe index is zero-based."
            assert self.diagrams[prior].angles[connect[0]] is not None, \
                   "The connection cannot be made.  Check that the magnitude " \
                   "of flow " + str(connect[0]) + " of diagram " + str(prior) \
                   + "is greater than or equal to the specified tolerance."
            flow_error = self.diagrams[prior].flows[connect[0]] \
                         + flows[connect[1]]
            # Without namedtuple:
            #assert self.diagrams[prior][2][connect[0]] is not None, \
            #       "The connection cannot be made.  Check that the magnitude " \
            #       "of flow " + str(connect[0]) + " of diagram " + str(prior) \
            #       + "is greater than or equal to the specified tolerance."
            #flow_error = self.diagrams[prior][1][connect[0]] \
            #             + flows[connect[1]]
            assert abs(flow_error) < self.tolerance, \
                   "The scaled sum of the connected flows is " \
                   + str(flow_error) + ", which is not within the tolerance (" \
                   + str(self.tolerance) + ")."

        # Determine if the flows are inputs.
        are_inputs = [None]*n
        for i, flow in enumerate(flows):
            if flow >= self.tolerance:
                are_inputs[i] = True
            elif flow <= -self.tolerance:
                are_inputs[i] = False
            else:
                verbose.report("The magnitude of flow " + str(i) + " (" \
                      + str(flow) + ") is below the tolerance (" \
                      + str(self.tolerance) + ").\nIt will not be shown, and " \
                      + "it cannot be used in a connection.", 'helpful')

        # Determine the angles of the arrows (before rotation).
        angles = [None]*n
        for i, (orient, is_input) in enumerate(zip(orientations, are_inputs)):
            if orient == 1:
                if is_input:
                    angles[i] = DOWN
                elif is_input == False:  # Be specific since is_input can be None.
                    angles[i] = UP
            elif orient == 0:
                if is_input is not None:
                    angles[i] = RIGHT
            else:
                assert orient == -1, \
                       "The value of orientations[" + str(i) + "] is " \
                       + str(orient) + ", but it must be -1, 0, or 1."
                if is_input:
                    angles[i] = UP
                elif is_input == False:
                    angles[i] = DOWN

        # Justify the lengths of the paths.
        if iterable(pathlengths):
            assert len(pathlengths) == n, \
                "If pathlengths is a list, then pathlengths and flows must " \
                "have the same length.\npathlengths has length " \
                + str(len(pathlengths)) + ", but flows has length " + str(n) + "."
        else:  # Make pathlengths into a list.
            urlength = pathlengths
            ullength = pathlengths
            lrlength = pathlengths
            lllength = pathlengths
            d = dict(RIGHT=pathlengths)
            pathlengths = [d.get(angle, 0) for angle in angles]
            # Determine the lengths of the top-side arrows from the middle outwards.
            for i, (angle, is_input, flow) \
                in enumerate(zip(angles, are_inputs, scaled_flows)):
                if angle == DOWN and is_input:
                    pathlengths[i] = ullength
                    ullength += flow
                elif angle == UP and not is_input:
                    pathlengths[i] = urlength
                    urlength -= flow  # Flow is negative for outputs
            # Determine the lengths of the bottom-side arrows from the middle outwards.
            for i, (angle, is_input, flow) \
                in enumerate(zip(angles, are_inputs, scaled_flows)[::-1]):
                if angle == UP and is_input:
                    pathlengths[n-i-1] = lllength
                    lllength += flow
                elif angle == DOWN and not is_input:
                    pathlengths[n-i-1] = lrlength
                    lrlength -= flow
            # Determine the lengths of the left-side arrows from the bottom upwards.
            has_left_input = False
            for i, (angle, is_input, spec) \
                in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))[::-1]):
                if angle == RIGHT:
                    if is_input:
                        if has_left_input:
                            pathlengths[n-i-1] = 0
                        else:
                            has_left_input = True
            # Determine the lengths of the right-side arrows from the top downwards.
            has_right_output = False
            for i, (angle, is_input, spec) \
                in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))):
                if angle == RIGHT:
                    if not is_input:
                        if has_right_output:
                            pathlengths[i] = 0
                        else:
                            has_right_output = True

        # Begin the subpaths, and smooth the transition if the sum of the flows
        # is nonzero.
        urpath = [(Path.MOVETO, [(self.gap - trunklength / 2.0),   # Upper right
                                 gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
                                 gain / 2.0]),
                  (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
                                 gain / 2.0]),
                  (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
                                 -loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
                                 -loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap),
                                 -loss / 2.0])]
        llpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),   # Lower left
                                 loss / 2.0]),
                  (Path.LINETO, [(trunklength / 2.0 - self.gap) / 2.0,
                                 loss / 2.0]),
                  (Path.CURVE4, [(trunklength / 2.0 - self.gap) / 8.0,
                                 loss / 2.0]),
                  (Path.CURVE4, [(self.gap - trunklength / 2.0) / 8.0,
                                 -gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0) / 2.0,
                                 -gain / 2.0]),
                  (Path.LINETO, [(self.gap - trunklength / 2.0),
                                 -gain / 2.0])]
        lrpath = [(Path.LINETO, [(trunklength / 2.0 - self.gap),   # Lower right
                                 loss / 2.0])]
        ulpath = [(Path.LINETO, [self.gap - trunklength / 2.0,     # Upper left
                                 gain / 2.0])]

        # Add the subpaths and assign the locations of the tips and labels.
        tips = np.zeros((n,2))
        label_locations = np.zeros((n,2))
        # Add the top-side inputs and outputs from the middle outwards.
        for i, (angle, is_input, spec) \
            in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))):
            if angle == DOWN and is_input:
                tips[i,:], label_locations[i,:] = self._add_input(ulpath, angle, *spec)
            elif angle == UP and not is_input:
                tips[i,:], label_locations[i,:] = self._add_output(urpath, angle, *spec)
        # Add the bottom-side inputs and outputs from the middle outwards.
        for i, (angle, is_input, spec) \
            in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))[::-1]):
            if angle == UP and is_input:
                tips[n-i-1,:], label_locations[n-i-1,:] = self._add_input(llpath, angle, *spec)
            elif angle == DOWN and not is_input:
                tips[n-i-1,:], label_locations[n-i-1,:] = self._add_output(lrpath, angle, *spec)
        # Add the left-side inputs from the bottom upwards.
        has_left_input = False
        for i, (angle, is_input, spec) \
            in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))[::-1]):
            if angle == RIGHT and is_input:
                if not has_left_input:
                    # Make sure the lower path extends at least as far as the upper one.
                    if llpath[-1][1][0] > ulpath[-1][1][0]:
                        llpath.append((Path.LINETO, [ulpath[-1][1][0], llpath[-1][1][1]]))
                    has_left_input = True
                tips[n-i-1,:], label_locations[n-i-1,:] = self._add_input(llpath, angle, *spec)
        # Add the right-side outputs from the top downwards.
        has_right_output = False
        for i, (angle, is_input, spec) \
            in enumerate(zip(angles, are_inputs, zip(scaled_flows, pathlengths))):
            if angle == RIGHT and not is_input:
                if not has_right_output:
                    # Make sure the upper path extends at least as far as the lower one.
                    if urpath[-1][1][0] < lrpath[-1][1][0]:
                        urpath.append((Path.LINETO, [lrpath[-1][1][0], urpath[-1][1][1]]))
                    has_right_output = True
                tips[i,:], label_locations[i,:] = self._add_output(urpath, angle, *spec)
        # Trim any hanging vertices.
        if not has_left_input:
            ulpath.pop()
            llpath.pop()
        if not has_right_output:
            lrpath.pop()
            urpath.pop()

        # Concatenate the subpaths in the correct order (clockwise from top).
        path = urpath + self._revert(lrpath) + llpath + self._revert(ulpath) \
               + [(Path.CLOSEPOLY, urpath[0][1])]

        # Create a patch with the Sankey outline.
        codes, vertices = zip(*path)
        vertices = np.array(vertices)
        def _get_angle(a, r):
            if a is None: return None
            else: return a + r

        if prior is None:
            if rotation != 0: # By default, none of this is needed.
                angles = [_get_angle(angle, rotation) for angle in angles]
                rotate = Affine2D().rotate_deg(rotation*90).transform_point
                tips = rotate(tips)
                label_locations = rotate(label_locations)
                vertices = rotate(vertices)
            text = self.ax.text(0, 0, s=patchlabel, ha='center', va='center')
        else:
            rotation = self.diagrams[prior].angles[connect[0]]  - angles[connect[1]]
            # Without namedtuple:
            #rotation = self.diagrams[prior][2][connect[0]] \
            #           - angles[connect[1]]
            angles = [_get_angle(angle, rotation) for angle in angles]
            rotate = Affine2D().rotate_deg(rotation*90).transform_point
            tips = rotate(tips)
            offset = self.diagrams[prior].tips[connect[0]] - tips[connect[1]]
            # Without namedtuple:
            #offset = self.diagrams[prior][3][connect[0]] - tips[connect[1]]
            translate = Affine2D().translate(*offset).transform_point
            tips = translate(tips)
            label_locations = translate(rotate(label_locations))
            vertices = translate(rotate(vertices))
            kwds = dict(s=patchlabel, ha='center', va='center')
            text = self.ax.text(*offset, **kwds)
        if False:                           # DEBUG
            print "llpath\n", llpath
            print "ulpath\n", self._revert(ulpath)
            print "urpath\n", urpath
            print "lrpath\n", self._revert(lrpath)
            xs, ys = zip(*vertices)
            self.ax.plot(xs, ys, 'go-')
        patch = PathPatch(Path(vertices, codes),
                          fc=kwargs.pop('fc', kwargs.pop('facecolor', # Custom
                                        '#bfd1d4')),                  # defaults
                          lw=kwargs.pop('lw', kwargs.pop('linewidth',
                                        '0.5')),
                          **kwargs)
        self.ax.add_patch(patch)

        # Add the path labels.
        for i, (number, angle) in enumerate(zip(flows, angles)):
            if labels[i] is None or angle is None:
                labels[i] = ''
            elif self.unit is not None:
                quantity = self.format%abs(number) + self.unit
                if labels[i] != '':
                    labels[i] += "\n"
                labels[i] += quantity
        texts = []
        for i, (label, location) in enumerate(zip(labels, label_locations)):
            if label: s = label
            else: s = ''
            texts.append(self.ax.text(x=location[0], y=location[1],
                                      s=s,
                                      ha='center', va='center'))
        # Text objects are placed even they are empty (as long as the magnitude
        # of the corresponding flow is larger than the tolerance) in case the
        # user wants to provide labels later.

        # Expand the size of the diagram if necessary.
        self.extent = (min(np.min(vertices[:,0]), np.min(label_locations[:,0]), self.extent[0]),
                       max(np.max(vertices[:,0]), np.max(label_locations[:,0]), self.extent[1]),
                       min(np.min(vertices[:,1]), np.min(label_locations[:,1]), self.extent[2]),
                       max(np.max(vertices[:,1]), np.max(label_locations[:,1]), self.extent[3]))
        # Include both vertices _and_ label locations in the extents; there are
        # where either could determine the margins (e.g., arrow shoulders).

        # Add this diagram as a subdiagram.
        self.diagrams.append(Bunch(patch=patch, flows=flows, angles=angles,
                                        tips=tips, text=text, texts=texts))
        # Without namedtuple:
        #self.diagrams.append((patch, flows, angles, tips, text, texts))

        # Allow a daisy-chained call structure (see docstring for the class).
        return self

    def finish(self):
        """Adjust the axes and return a list of information about the
        subdiagram(s).

        Each entry in the subdiagram list is a namedtuple with the following
        fields:
        patch:   Sankey outline (an instance of maplotlib.patches.PathPatch)
        flows:   Values of the flows (positive for input, negative for output)
        angles:  List of angles of the arrows [deg/90]
                 For example, if the diagram has not been rotated, an input to
                 the top side will have an angle of 3 (DOWN), and an output from
                 the top side will have an angle of 1 (UP). If a flow has been
                 skipped (because it is too close to 0), then its angle will be
                 None.
        tips:    Array where each row is an [x, y] pair indicating the positions
                 of the tips (or "dips") of the flow paths
                 If the magnitude of a flow is less the tolerance for the Sankey
                 class, the flow is skipped and its tip will be at the center of
                 the diagram.
        text:    matplotlib.text.Text instance for the label of the diagram
        texts:   List of matplotlib.text.Text instances for the labels of flows
        """
        self.ax.axis([self.extent[0] - self.margin,
                      self.extent[1] + self.margin,
                      self.extent[2] - self.margin,
                      self.extent[3] + self.margin])
        self.ax.set_aspect('equal', adjustable='datalim')
        return self.diagrams

    def __init__(self, ax=None, scale=1.0, unit='', format='%G', gap=0.25,
                 radius=0.1, shoulder=0.03, offset=0.15, head_angle=100,
                 margin=0.4, tolerance=1e-6, **kwargs):
        """Create a new Sankey diagram.

        ax:           Axes onto which the data should be plotted
                      If not provided, they will be created.
        scale:        Scaling factor for the flows
                      This factor sizes the width of the paths in order to
                      maintain proper layout.  The same scale is applied to all
                      subdiagrams.  The value should be chosen such that the
                      product of the scale and the sum of the inputs is
                      approximately 1 (and the product of the scale and the sum
                      of the outputs is approximately -1).
        unit:         Unit associated with the flow quantities
                      If unit is None, then none of the quantities are labeled.
        format:       A Python number formatting string to be used in labeling
                      the flow as a quantity (i.e., a number times a unit, where
                      the unit is given)
        gap:          Space between paths in that break in/break away to/from
                      the top or bottom
        radius:       Inner radius of the vertical paths
        shoulder:     Output arrow shoulder
        offset:       Text offset (away from the dip or tip of the arrow)
        head_angle:   Angle of the arrow heads (and negative of the angle of the
                      tails) [deg]
        margin:       Minimum space between Sankey outlines and the edge of the
                      plot area
        tolerance:    Acceptable maximum of the magnitude of the sum of flows
                      The magnitude of the sum of connected flows cannot be
                      greater than this value.  If the magnitude of the sum of
                      the flows of a subdiagram is greater than this value, a
                      warning is displayed.
        **kwargs:     Propagated to Sankey.add()

        The above arguments are applied to all subdiagrams so that there is
        consistent alignment and formatting.

        If this class is instantiated with any keyworded arguments (**kwargs)
        other than those explicitly listed above, they will be passed to the
        add() method, which will create the first subdiagram.

        In order to draw a complex Sankey diagram, create an instance of this
        class by calling it without any **kwargs:
            sankey = Sankey()
        Then add simple Sankey sub-diagrams:
            sankey.add() # 1
            sankey.add() # 2
            #...
            sankey.add() # n
        Finally, create the full diagram:
            sankey.finish()
        Or, instead, simply daisy-chain those calls:
            Sankey().add().add...  .add().finish()
        """
        # Check the arguments.
        assert gap >= 0, \
               "The gap is negative.\n" \
               + "This isn't allowed because it would cause the paths to overlap."
        assert radius <= gap, \
               "The inner radius is greater than the path spacing.\n" \
               + "This isn't allowed because it would cause the paths to overlap."
        assert head_angle >= 0, \
               "The angle is negative.\n" \
               + "This isn't allowed because it would cause inputs to look "\
               + "like outputs and vice versa."
        assert tolerance >= 0, \
               "The tolerance is negative.\nIt must be a magnitude."

        # Create axes if necessary.
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

        self.diagrams = []

        # Store the inputs.
        self.ax = ax
        self.unit = unit
        self.format = format
        self.scale = scale
        self.gap = gap
        self.radius = radius
        self.shoulder = shoulder
        self.offset = offset
        self.margin = margin
        self.pitch = np.tan(np.pi * (1 - head_angle / 180.0) / 2.0)
        self.tolerance = tolerance

        # Initialize the vertices of tight box around the diagram(s).
        self.extent = np.array((np.inf, -np.inf, np.inf, -np.inf))

        # If there are any kwargs, create the first subdiagram.
        if len(kwargs):
            self.add(**kwargs)

if __name__ == '__main__':
    """Demonstrate the Sankey class.
    """
    import matplotlib.pyplot as plt
    from itertools import cycle

    # Example 1 -- Mostly defaults
    # This demonstrates how to create a simple diagram by implicitly calling the
    # Sankey.add() method and by appending finish() to the call to the class.
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
    #   8. Changing the appearance of the patch and the labels after the figure
    #      is created
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
               alpha=0.2, lw=2.0)  # Arguments to matplotlib.patches.PathPatch()
    diagrams = sankey.finish()
    diagrams[0].patch.set_facecolor('#37c959')
    diagrams[0].texts[-1].set_color('r')
    diagrams[0].text.set_fontweight('bold')
    # Without namedtuple:
    #diagrams[0][0].set_facecolor('#37c959')
    #diagrams[0][5][-1].set_color('r')
    #diagrams[0][4].set_fontweight('bold')
    # Notice:
    #   1. Since the sum of the flows isn't zero, the width of the trunk isn't
    #      uniform.  A message is given in the terminal window.
    #   2. The second flow doesn't appear because its value is zero.  A messsage
    #      is given in the terminal window.

    # Example 3
    # This demonstrates:
    #   1. Connecting two systems
    #   2. Turning off the labels of the quantities
    #   3. Adding a legend
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
    # Without namedtuple:
    #diagrams[-1][0].set_hatch('/')

    plt.legend(loc='best')
    # Notice that only one connection is specified, but the systems form a
    # circuit since: (1) the lengths of the paths are justified and (2) the
    # orientation and ordering of the flows is mirrored.

    # Example 4
    # This tests a long chain of connections.
    links_per_side = 6
    def side(sankey, n=1):
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
        prior = len(sankey.diagrams)
        sankey.add(flows=[1, -1], orientations=[0, 1],
                   patchlabel=str(prior), facecolor='k',
                   prior=prior-1, connect=(1, 0), alpha=0.5)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                         title="Why would you want to do this?" \
                               "\n(But you could.)")
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
    # 2. The first diagram is rotated 45 degrees, so all other diagrams are
    #    rotated accordingly.

    # Example 5
    # This demonstrates a practical example -- a Rankine power cycle.
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                         title="Rankine Power Cycle: Example 8.6 from Moran and Shapiro\n"
                               + "\x22Fundamentals of Engineering Thermodynamics\x22, 6th ed., 2008")
    Hdot = np.array([260.431, 35.078, 180.794, 221.115, 22.700,
            142.361, 10.193, 10.210, 43.670, 44.312,
            68.631, 10.758, 10.758, 0.017, 0.642,
            232.121, 44.559, 100.613, 132.168])*1.0e6 # W
    sankey = Sankey(ax=ax, format='%.3G', unit='W', gap=0.5, scale=1.0/Hdot[0])
    # Shared copy:
    #Hdot = [260.431, 35.078, 180.794, 221.115, 22.700,
    #        142.361, 10.193, 10.210, 43.670, 44.312,
    #        68.631, 10.758, 10.758, 0.017, 0.642,
    #        232.121, 44.559, 100.613, 132.168] # MW
    #sankey = Sankey(ax=ax, format='%.3G', unit=' MW', gap=0.5, scale=1.0/Hdot[0])
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
        # Without namedtuple:
        #diagram[4].set_fontweight('bold')
        #diagram[4].set_fontsize('10')
        #for text in diagram[5]:
            text.set_fontsize('10')
    # Notice that the explicit connections are handled automatically, but the
    # implicit ones currently are not.  The lengths of the paths and the trunks
    # must be adjusted manually, and that is a bit tricky.

    plt.show()

