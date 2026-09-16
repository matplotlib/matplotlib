from matplotlib import cbook
from .artist import Artist
from .transforms import Bbox

class Pie(Artist):
	"""
	Compound Artist representing a pie chart.

	.. versionadded:: 3.12

	Attributes
	----------
	wedges : list of `~matplotlib.patches.Wedge`
		The artists of the pie wedges.

	values : `numpy.ndarray`
		The data that the pie is based on.

	fracs : `numpy.ndarray`
		The fraction of the pie that each wedge represents.

	texts : list of list of `~matplotlib.text.Text`
		The artists of any labels on the pie wedges. Each inner list has one
		text label per wedge.
	"""

	def __init__(self, wedges, values, normalize, shadows=None):
		"""
		Parameters
		----------
		wedges : list of `~matplotlib.patches.Wedge`
			The artists of the pie wedges.
		values : `numpy.ndarray`
			The data that the pie is based on.
		normalize : bool, default: True
			Whether the pie slices are normalized to sum to 1.
		shadows : list of `~matplotlib.patches.Shadow`, optional
			Shadow patches associated with the wedges.
		"""
		super().__init__()
		self.set_clip_on(False)
		self.wedges = wedges
		self._texts = []
		self._values = values
		self._normalize = normalize
		self._shadows = list(shadows) if shadows else []

	@property
	def texts(self):
		# Only return non-empty sublists.  An empty sublist may have been added
		# for backwards compatibility of the Axes.pie return value (see __getitem__).
		return [t_list for t_list in self._texts if t_list]

	@property
	def values(self):
		result = self._values.copy()
		result.flags.writeable = False
		return result

	@property
	def fracs(self):
		if self._normalize:
			result = self._values / self._values.sum()
		else:
			result = self._values

		result.flags.writeable = False
		return result

	def add_texts(self, texts):
		"""Add a list of `~matplotlib.text.Text` objects to the pie artist."""
		self._texts.append(texts)
		fig = self.get_figure(root=False)
		for t in texts:
			t.set_figure(fig)
			t.axes = self.axes
			if not t.is_transform_set():
				t.set_transform(self.get_transform())

	def remove(self):
		super().remove()

	def __getitem__(self, key):
		# needed to support unpacking into a tuple for backward compatibility of the
		# Axes.pie return value
		return (self.wedges, *self._texts)[key]

	def draw(self, renderer):
		if not self.get_visible():
			return
		renderer.open_group('pie', gid=self.get_gid())
		for s in self._shadows:
			s.draw(renderer)
		for w in self.wedges:
			w.draw(renderer)
		for t_list in self._texts:
			for t in t_list:
				t.draw(renderer)
		renderer.close_group('pie')
		self.stale = False

	def get_children(self):
		"""Return the Artists contained by the pie."""
		return [*self._shadows, *self.wedges, *cbook.flatten(self._texts)]

	def get_tightbbox(self, renderer=None):
		# docstring inherited
		if renderer is None:
			renderer = self.get_figure(root=True)._get_renderer()
		bboxes = [bbox for child in self.get_children()
				  if (bbox := child.get_tightbbox(renderer)) is not None
				  and bbox._is_finite()]
		return Bbox.union(bboxes) if bboxes else None

	@Artist.axes.setter
	def axes(self, new_axes):
		Artist.axes.fset(self, new_axes)
		for s in self._shadows:
			s.axes = new_axes
		for w in self.wedges:
			w.axes = new_axes
		for t_list in self._texts:
			for t in t_list:
				t.axes = new_axes

	def set_transform(self, t):
		super().set_transform(t)
		for s in self._shadows:
			s.set_transform(t)
		for w in self.wedges:
			w.set_transform(t)
		for t_list in self._texts:
			for txt in t_list:
				txt.set_transform(t)

	def set_figure(self, fig):
		super().set_figure(fig)
		for s in self._shadows:
			s.set_figure(fig)
		for w in self.wedges:
			w.set_figure(fig)
		for t_list in self._texts:
			for t in t_list:
				t.set_figure(fig)
