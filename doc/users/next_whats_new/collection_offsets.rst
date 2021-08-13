Setting collection offset transform after initialization
--------------------------------------------------------
`.collections.Collection.set_offset_transform()` was added.

Previously the offset transform could not be set after initialization. This can be helpful when creating a `.collections.Collection` outside an axes object and later adding it with `.Axes.add_collection()` and settings the offset transform to `.Axes.transData`.
