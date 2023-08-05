Method parameters renamed to match base classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The only parameter of ``transform_affine`` and ``transform_non_affine`` in ``Transform`` subclasses is renamed
to *values*.

The *points* parameter of ``transforms.IdentityTransform.transform`` is renamed to *values*.

The *trans* parameter of ``table.Cell.set_transform`` is renamed to *t* consistently with
`.Artist.set_transform`.

The *clippath* parameters of ``axis.Axis.set_clip_path``  and ``axis.Tick.set_clip_path`` are
renamed to *path* consistently with `.Artist.set_clip_path`.

The *s* parameter of ``images.NonUniformImage.set_filternorm`` is renamed to *filternorm*
consistently with ```_ImageBase.set_filternorm``.

The *s* parameter of ``images.NonUniformImage.set_filterrad`` is renamed to *filterrad*
consistently with ```_ImageBase.set_filterrad``.
