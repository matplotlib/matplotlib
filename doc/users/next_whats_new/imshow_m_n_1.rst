Imshow now coerces 3D arrays with depth 1 to 2D
------------------------------------------------
Starting from this version arrays of size MxNx1 will be coerced into MxN 
for displaying. This means commands like ``plt.imshow(np.random.rand(3, 3, 1))`` 
will no longer return an error message that the image shape is invalid.
