Setting collection offset transform after initialization
--------------------------------------------------------
`~matplotlib.collections.Collection.set_offset_transform` was added.

Previously the offset transform could not be set after initialization. This can be helpful when creating a `Collection` outside an axes object and later adding it with `ax.add_collection` and settings the offset transform to `ax.transData`.
