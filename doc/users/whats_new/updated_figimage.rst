updated figimage to take optional resize parameter
----------------------------------------------------
Added the ability to plot simple 2D-Array using plt.figimage(X, resize=True).
This is useful for plotting simple 2D-Array without the Axes or whitespacing
around the image.
Example:
	data = np.random.random( [500, 500] )
	plt.figimage(data, resize=True)
