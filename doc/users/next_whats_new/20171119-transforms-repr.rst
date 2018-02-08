Improved `repr` for `Transform`\s
---------------------------------

`Transform`\s now indent their `repr`\s in a more legible manner:

.. code-block:: ipython

   In [1]: l, = plt.plot([]); l.get_transform()
   Out[1]: 
   CompositeGenericTransform(
      TransformWrapper(
         BlendedAffine2D(
               IdentityTransform(),
               IdentityTransform())),
      CompositeGenericTransform(
         BboxTransformFrom(
               TransformedBbox(
                  Bbox(x0=-0.05500000000000001, y0=-0.05500000000000001, x1=0.05500000000000001, y1=0.05500000000000001),
                  TransformWrapper(
                     BlendedAffine2D(
                           IdentityTransform(),
                           IdentityTransform())))),
         BboxTransformTo(
               TransformedBbox(
                  Bbox(x0=0.125, y0=0.10999999999999999, x1=0.9, y1=0.88),
                  BboxTransformTo(
                     TransformedBbox(
                           Bbox(x0=0.0, y0=0.0, x1=6.4, y1=4.8),
                           Affine2D(
                              [[ 100.    0.    0.]
                              [   0.  100.    0.]
                              [   0.    0.    1.]])))))))
