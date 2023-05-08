3D plots can share view angles
------------------------------

3D plots can now share the same view angles, so that when you rotate one plot
the other plots also rotate. This can be done with the *shareview* keyword
argument when adding an axes, or by using the *ax1.shareview(ax2)* method of
existing 3D axes.
