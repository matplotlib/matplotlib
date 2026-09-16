from matplotlib import cbook
from .artist import Artist

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

	def remove(self):
		"""Remove all wedges and texts from the axes"""
		for artist_list in self.wedges, self._texts:
			for artist in cbook.flatten(artist_list):
				artist.remove()

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
