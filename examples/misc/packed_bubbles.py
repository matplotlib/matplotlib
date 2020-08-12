"""
Create a packed bubble / non overlapping bubble chart to represent scalar data.
The presented algorithm tries to move all bubbles as close to the center of
mass as possible while avoiding some collisions by moving aroud colliding
objects. In this example we plot the market share of different desktop
browsers.
"""

import numpy as np
import matplotlib.pyplot as plt

browser_market_share = {
    'browsers': ['firefox', 'chrome', 'safari', 'edge', 'ie', 'opera'],
    'market_share': [783, 6881, 371, 704, 587, 124],
    'color': ['b', 'g', 'r', 'c', 'm', 'y']
}


class BubbleChart:
    def __init__(self, r=None, a=None, bubble_distance=0):
        """
        setup for bubble collapse

        Parameters
        ----------
        r : list, optional
            radius of the bubbles. Defaults to None.
        a : list, optional
            area of the bubbles. Defaults to None.
        bubble_distance : int, optional
            minimal distance the bubbles should
            have after collapsing. Defaults to 0.

        Notes
        -----
        If r or a is sorted, the results might look weird
        """
        if r is None:
            r = np.sqrt(a / np.pi)
        if a is None:
            a = np.power(r, 2) * 2

        self.bubble_distance = bubble_distance
        self.n = len(r)
        self.bubbles = np.ones((len(self), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = a
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_distance
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self)))
        grid = np.arange(0, length * self.maxstep, self.maxstep)
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self)]
        self.bubbles[:, 1] = gy.flatten()[:len(self)]

        self.com = self.center_of_mass()

    def __len__(self):
        return self.n

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_distance

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass

        Parameters
        ----------
        n_iterations :int, optional
            number of moves to perform. Defaults to 50.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towoards the center of mass
                # dir_vec from bubble to com
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten dir_vec to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate dir vec
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthagonal vec
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which dir to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        draw the bubble plot

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
        labels : list
            labels of the bubbles
        colors : list
            colors of the bubbles
        """
        for i in range(len(self)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')


# set market share of the browsers as area of the bubbles
bubble_plot = BubbleChart(a=np.array(
    browser_market_share['market_share']), bubble_distance=1)

bubble_plot.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
bubble_plot.plot(
    ax, browser_market_share['browsers'], browser_market_share['color'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('Browser market share')

plt.show()
