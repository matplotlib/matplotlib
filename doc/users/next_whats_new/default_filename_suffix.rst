Stop adding a suffix to suggest unique file name
------------------------------------------------

Previously, when saving a figure to a file using the GUI's
save dialog box, if the default filename (based on the
figure window title) already existed on disk, Matplotlib
would append a suffix (e.g. `Figure_1-1.png`), preventing
the dialog from prompting to overwrite the file. This
behaviour has been removed. Now if the file name exists on
disk, the user is prompted whether or not to overwrite it.
This eliminates guesswork, and allows intentional
overwriting, especially when the figure name has been
manually set using `fig.canvas.set_window_title()`.
