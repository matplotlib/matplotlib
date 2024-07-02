Added support for Non 2-dimensional transforms
----------------------------------------------

Support has been added for transforms in matplotlib that aren't 2D.

``AffineImmutable`` directly replaces ``Affine2DBase``, and introduces a ``dims``
keyword that specifies the dimension of the transform, defaulting to 2.

``BlendedAffine`` directly replaces ``BlendedAffine2D``, and can blend more than
two transforms, with each transform handling a different axis.

``CompositeAffine`` directly replaces ``CompositeAffine2D``, and composes two Affine
transforms, as long as they have the same dimensions.

``IdentityTransform`` can create identity matrices of any dimension, through the use of
the ``dims`` keyword.
