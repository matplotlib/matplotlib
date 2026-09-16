import matplotlib.pyplot as plt


def test_pie_remove():
	fig, ax = plt.subplots()
	pie = ax.pie([2, 3], wedge_labels=['foo', 'bar'], autopct="%1.0f%%")
	ax.pie_label(pie, ['baz', 'qux'])

	assert len(ax.patches) == 0
	assert pie in ax._children
	# We have added 6 labels but pie also adds an empty Text artist to each
	# wedge if labeldistance is not None and labels is not passed
	assert len(ax.texts) == 8

	pie.remove()
	assert pie not in ax._children
	# assert not ax.texts


def test_pie_unpack_backcompat():
	fig, ax = plt.subplots()
	wedges, texts, autotexts = ax.pie(
		[2, 3], labels=['foo', 'bar'], autopct="%1.0f%%", labeldistance=None)

	assert len(wedges) == 2
	assert isinstance(texts, list)
	assert not texts
	assert len(autotexts) == 2
