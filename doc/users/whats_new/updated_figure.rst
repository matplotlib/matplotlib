Updated Figure.savefig()
------------------------

Added support to save the figure with the same dpi as the figure on the screen using dpi='figure'

Example:
	f = plt.figure(dpi=25)				# dpi set to 25
	S = plt.scatter([1,2,3],[4,5,6])
	f.savefig('output.png', dpi='figure')	# output savefig dpi set to 25 (same as figure)
