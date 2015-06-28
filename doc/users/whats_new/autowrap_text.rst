Auto-wrapping Text
------------------
Added the keyword argument "wrap" to Text, which automatically breaks long lines of text when being drawn.
Works for any rotated text, different modes of alignment, and for text that are either labels or titles.

Example ::

    plt.text(1, 1,
             "This is a really long string that should be wrapped so that "
	     "it does not go outside the figure.", wrap=True)
