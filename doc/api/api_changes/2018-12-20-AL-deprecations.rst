Deprecations
````````````

Multiple internal functions that were exposed as part of the public API
of ``mpl_toolkits.mplot3d`` are deprecated,

**mpl_toolkits.mplot3d.art3d**

- :func:`mpl_toolkits.mplot3d.art3d.norm_angle`
- :func:`mpl_toolkits.mplot3d.art3d.norm_text_angle`
- :func:`mpl_toolkits.mplot3d.art3d.path_to_3d_segment`
- :func:`mpl_toolkits.mplot3d.art3d.paths_to_3d_segments`
- :func:`mpl_toolkits.mplot3d.art3d.path_to_3d_segment_with_codes`
- :func:`mpl_toolkits.mplot3d.art3d.paths_to_3d_segments_with_codes`
- :func:`mpl_toolkits.mplot3d.art3d.get_patch_verts`
- :func:`mpl_toolkits.mplot3d.art3d.get_colors`
- :func:`mpl_toolkits.mplot3d.art3d.zalpha`

**mpl_toolkits.mplot3d.proj3d**

- :func:`mpl_toolkits.mplot3d.proj3d.line2d`
- :func:`mpl_toolkits.mplot3d.proj3d.line2d_dist`
- :func:`mpl_toolkits.mplot3d.proj3d.line2d_seg_dist`
- :func:`mpl_toolkits.mplot3d.proj3d.mod`
- :func:`mpl_toolkits.mplot3d.proj3d.proj_transform_vec`
- :func:`mpl_toolkits.mplot3d.proj3d.proj_transform_vec_clip`
- :func:`mpl_toolkits.mplot3d.proj3d.vec_pad_ones`
- :func:`mpl_toolkits.mplot3d.proj3d.proj_trans_clip_points`

If your project relies on these functions, consider vendoring them.
