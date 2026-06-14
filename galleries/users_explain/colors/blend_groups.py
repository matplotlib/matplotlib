"""
.. _blend-groups:

==========================================
Blending and compositing groups of artists
==========================================

An advanced technique of blending artists (see :ref:`blend-modes`) is to use a
blend group, also known as a transparency group.  Blend groups can be isolated,
knockout, or both:

* An isolated group has the artists within the group rendered into a separate
  buffer, and the result is subsequently blended into the primary buffer.
* A knockout group has the artists within the group blended onto the initial
  backdrop, with each successive artist ignoring any modifications underneath it
  by preceding artists in the group.

The methods to open and close groups are found on the backend renderer, but
user code does not typically directly access the renderer.  The convenience
class below (``ArtistGroup``) makes it straightforward to form a blend group
from a list of artists.  Specifying ``group_blend_mode`` to something other than
``None`` makes the blend group an isolated group.  Specifying ``knockout=True``
makes the blend group a knockout group.

The example below shows:

* The left panel shows the behavior of a blend group that is neither isolated
  nor knockout.  The result is the same as not using a blend group at all.  A
  yellow circle and a green circle are successively blended with the "multiply"
  blend mode into the backdrop.
* The middle panel shows how the behavior changes when the two circles are in an
  isolated blend group.  The yellow circle is rendered into an isolated buffer,
  so its "multiply" blend mode has no visible effect.  The green circle is then
  blended with the yellow circle using "multiply".  Finally, the isolated buffer
  is blended into the primary buffer using "normal".  Thus, the "multiply" blend
  mode affects only the overlap between the two circles, and does not interact
  with the backdrop at all due to the isolation.
* The right panel shows how the behavior changes when the blend group is both
  isolated and knockout.  The green circle knocks out the portion of the yellow
  circle that is overlapped.  Since there is no longer any overlapping elements
  in the isolated buffer, the blend modes within the group have no visible
  effect.  As before, the isolated buffer is then blended into the primary
  buffer using "normal".

Support for the different types of blend groups depends on the backend.  See the
table at the bottom for details.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.artist import Artist
from matplotlib.patches import Circle


class ArtistGroup(Artist):
    def __init__(self, artist_list, *,
                 group_blend_mode=None, group_alpha=1, knockout=False):
        self._artist_list = artist_list
        self._group_blend_mode = group_blend_mode
        self._group_alpha = group_alpha
        self._knockout = knockout
        super().__init__()

    def draw(self, renderer):
        renderer.open_blend_group(self._group_blend_mode, alpha=self._group_alpha,
                                  knockout=self._knockout)
        for a in self._artist_list:
            if not a.is_transform_set():
                a.set_transform(self.get_transform())
            if getattr(a, "axes", None) is None:
                a.axes = self.axes
            a.draw(renderer)
        renderer.close_blend_group()


fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')

for i, (group_blend_mode, knockout) in enumerate([(None, False),
                                                  ("normal", False),
                                                  ("normal", True)]):
    axs[i].set_xlim(-1, 1)
    axs[i].set_ylim(-1, 1)
    axs[i].set_aspect("equal")
    axs[i].set_axis_off()

    axs[i].imshow(np.arange(20*20).reshape((20, 20)) % 19,
                  cmap='Spectral', extent=[-1, 1, -1, 1])

    left = Circle((-0.25, 0), 0.6, fc='y', alpha=0.75, blend_mode='multiply')
    right = Circle((0.25, 0), 0.6, fc='g', alpha=0.75, blend_mode='multiply')

    both = ArtistGroup([left, right],
                       group_blend_mode=group_blend_mode, knockout=knockout)
    axs[i].add_artist(both)

axs[0].set_title('neither isolated nor knockout')
axs[1].set_title('isolated only')
axs[2].set_title('isolated and knockout')

plt.show()

# %%
#
# This table shows which types of blend groups are supported by each
# backend type (✅ = supported, ❌ = not supported).
#
# +--------------------+-----+-------+-----+-----+-----+---------+
# | Option             | Agg | Cairo | SVG | PDF | PGF | PS      |
# +====================+=====+=======+=====+=====+=====+=========+
# | neither isolated   | ✅  |  ✅   | ✅  | ✅  | ✅  | ✅ [#]_ |
# | nor knockout       |     |       |     |     |     |         |
# +--------------------+-----+-------+-----+-----+-----+---------+
# | isolated only      | ✅  |  ✅   | ✅  | ✅  | ✅  | ❌      |
# +--------------------+-----+-------+-----+-----+-----+---------+
# | isolated and       | ✅  |  ✅   | ❌  | ✅  | ✅  | ❌      |
# | knockout           |     |       |     |     |     |         |
# +--------------------+-----+-------+-----+-----+-----+---------+
# | knockout only [#]_ | ❌  |  ❌   | ❌  | ✅  | ✅  | ❌      |
# +--------------------+-----+-------+-----+-----+-----+---------+
#
# .. [#] groups are not supported, but adding artists without a group is equivalent
# .. [#] not depicted in the example above
