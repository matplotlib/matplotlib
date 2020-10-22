``QuiverKey`` properties are now modifiable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.QuiverKey` object returned by `.pyplot.quiverkey` and `.axes.Axes.quiverkey`
formerly saved various properties as attributes during initialization. However,
modifying these attributes may or may not have had an effect on the final result.

Now all such properties have getters and setters, and may be modified after creation:

- `.QuiverKey.X` -> `.QuiverKey.get_x` / `.QuiverKey.set_x` /
  `.QuiverKey.get_position` / `.QuiverKey.set_position`
- `.QuiverKey.Y` -> `.QuiverKey.get_y` / `.QuiverKey.set_y` /
  `.QuiverKey.get_position` / `.QuiverKey.set_position`
- `.QuiverKey.label` -> `.QuiverKey.get_label_text` / `.QuiverKey.set_label_text`
- `.QuiverKey.labelcolor` -> `.QuiverKey.get_label_color` / `.QuiverKey.set_label_color`
- `.QuiverKey.labelpos` -> `.QuiverKey.get_label_pos` / `.QuiverKey.set_label_pos`
- `.QuiverKey.labelsep` is now read-only as it used a different unit (pixels)
  than the constructor (inches), and was automatically overwritten;
  `.QuiverKey.get_labelsep` and `.QuiverKey.set_labelsep` have been added which
  use inches
